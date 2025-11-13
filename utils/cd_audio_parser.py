#!/usr/bin/env python
"""
This script transcribe and analyze cd task audios.
from Hazard and task name, it will find GT csv of correct answer and expected answering time period
Then extract the headphone audio based on the period, then transcribe.
Finary, it calculates the answer accuracy and latency

Usage:
python cd_audio_parser.py
"""

import argparse
import glob
import io
import json
import re
import warnings
from pathlib import Path

import Levenshtein as lev
import noisereduce as nr
import numpy as np
import pandas as pd
import soundfile as sf
import stable_whisper
import torch
import whisper
import yaml
from metaphone import doublemetaphone

# from audio_transcribe import extract_non_silent_segments, transcribe
from pydub import AudioSegment
from tqdm import tqdm

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


def bytes_to_numpy(audio_bytes):
    with io.BytesIO(audio_bytes) as f:
        audio, samplerate = sf.read(f)
    return audio, samplerate


def audiosegment2numpy(src: AudioSegment):
    samples = np.array(src.get_array_of_samples())
    if src.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples


def numpy2audiosegment(src: np.array):
    sample_rate = whisper.audio.SAMPLE_RATE
    audio_array = (src * np.iinfo(np.int16).max).astype(np.int16)
    audio = AudioSegment(
        audio_array.tobytes(), frame_rate=sample_rate, sample_width=audio_array.dtype.itemsize, channels=1
    )
    return audio


def misheard_converter(transcribed_text, misheard_dict):
    words = transcribed_text.split()
    corrected_words = []
    for word in words:
        # correc words if in dict
        corrected_words.append(misheard_dict.get(word, word))

    return " ".join(corrected_words)


def normalize_text(text, task):
    if pd.isna(text):
        return text
    # Make it lower case
    text = text.lower()
    # Remove punctuation marks such as dots, commas, question marks, and exclamation points
    text = re.sub(r"[.,!?]", "", text)
    # Multiple spaces into one space
    text = re.sub(r"\s+", " ", text)
    # Remove excess space on both edges
    text = text.strip()

    if task == "nback_task":
        misheard_dict = misheard_dict_nback
    elif task == "statement_task":
        misheard_dict = misheard_dict_statement
    else:
        raise NotImplementedError
    text = misheard_converter(text, misheard_dict)
    return text


def nonsilent_valid(x):
    if len(x) > 0:
        if x[0][0] > 0.7:  # assume answering took at least 0.7 sec, otherwise noise or last task answer
            return True
        else:
            if len(x) > 1:
                if x[1][0] > 0.7:  # first nonsilent was invalid but next is valid
                    return True
    return False


# Get pronouce pattern by Metaphone
def get_metaphone(word):
    # doublemetaphone returns 2 results, typically first one is important
    return doublemetaphone(word)[0]


# Calc Levenshtein distance, then return similality
def calculate_similarity(word1, word2):
    metaphone1 = get_metaphone(word1)
    metaphone2 = get_metaphone(word2)

    distance = lev.distance(metaphone1, metaphone2)

    # Calc Similality (1 is perfect match)
    return 1 - (distance / max(len(metaphone1), len(metaphone2)))


def calc_match_score(ans, gt, task):
    """
    nback:
        If gt number is included in answer words, score is 1.0. Otherwise, 0.0
    statement:
        On each gt words (e.g. james ball yes), evaluate the pronounce similality with
        all the answer words (e.g. james bow yes) use maximum similality score. Order is not cared.

        For the similality calculation, we use Metaphone and Levenshtein distance.
        Metaphone: encode based on pronounce. 'ball' -> 'PL', 'bow' -> 'P', 'boat' -> 'PT'
        Levenshtein distance: distance between texts
    """

    if task == "nback_task":
        ans_l = re.split(r"[ -]", ans)  # 3 4 -> [3,4]
        score = float(gt in ans_l)
        return score
    elif task == "statement_task":
        ans_l = re.split(r"[ -]", ans)  # ['james', 'bow', 'yes']
        gt_l = re.split(r"[ -]", gt)  # ['james', 'ball', 'yes']

        ans_l_m = [get_metaphone(word) for word in ans_l]  # ['JMS', 'P', 'AS']
        gt_l_m = [get_metaphone(word) for word in gt_l]  # ['JMS', 'PL', 'AS']
        # how many gt words are included in answer words
        similality_on_gt_words = []
        for gt_word_m in gt_l_m:
            similarity_l = [calculate_similarity(gt_word_m, ans_word_m) for ans_word_m in ans_l_m]
            max_similaliry = np.max(similarity_l)
            similality_on_gt_words.append(max_similaliry)
        score = np.sum(similality_on_gt_words) / len(similality_on_gt_words)
        return score
    else:
        raise NotImplementedError


# dict to correct transcription
misheard_dict_nback_inv = {
    "0": ["zero", "siro"],
    "1": ["one", "wham", "what", "wow"],
    "2": ["two", "to", "till", "true", "due", "chew"],
    "3": ["three", "free", "theory"],
    "4": ["four", "for", "whore", "or", "sour", "floor", "soar"],
    "5": ["five", "bye", "hi", "fine", "fives", "size", "file", "files"],
    "6": ["six", "fix", "sex"],
    "7": ["seven"],
    "8": ["eight", "eights", "hey", "age", "aids", "a", "wait", "date"],
    "9": ["nine", "nye", "night", "no", "nigh", "nein", "nāi", "wǎn", "dine"],
}
misheard_dict_nback = {val: k for k, vals in misheard_dict_nback_inv.items() for val in vals}

misheard_dict_statement_inv = {
    "no": ["now"],
}
misheard_dict_statement = {val: k for k, vals in misheard_dict_statement_inv.items() for val in vals}


class WhisperWrapper:
    def __init__(self, model_source="whisper"):
        # Load the Whisper model.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_source = model_source
        model_name = "large-v3"
        # model_name = "large"
        if model_source == "whisper":
            self.model = whisper.load_model(model_name, device=device)
        elif model_source == "stable_whisper":
            self.model = stable_whisper.load_model(model_name, device=device)
        print(f"Whisper model, {model_name}, from {model_source} is loaded.")


class CDParser:
    def __init__(
        self,
        whisper_instance,
        path_to_clean="/data/motion-simulator-logs/Processed/Clean",
        output_dir="/data/motion-simulator-logs/Processed/Features/transcript",
        write_all_audio_chunks=False,
        write_misheard_audio_chunks=True,
        path_filter=None,
    ):
        self.whisper_instance = whisper_instance  # whisper model
        self.path_to_clean = Path(path_to_clean)
        self.output_dir = Path(output_dir)
        self.write_all_audio_chunks = write_all_audio_chunks
        self.write_misheard_audio_chunks = write_misheard_audio_chunks
        self.offset_start = 0.1
        self.offset_end = 0.1  # [sec]. extract response with offset_end to allow delayed response
        self.audio_files = glob.glob(str(self.path_to_clean / "Participants/*/*/*/*/*/sim_bag/headphone_audio.mp3"))

        # filter the mp3 paths
        if path_filter is not None:
            self.audio_files = [f for f in self.audio_files if path_filter in f]

        self.cd_audio_root = Path("/home/kimimasatamura/git/internal/motion_sim_hmi_scenarios/cognitive_tasks/")
        self.mapper = self.set_mapper()

    def load_gt(self, hazard, task):
        # find GT csv
        gt = pd.read_csv(self.cd_audio_root / self.mapper[hazard][task])

        if task == "nback_task":
            # add start_response, end_response, expected_response to have same format with statement gt
            init_duration = 7.208  # length of "We will begin a 1-back test shortly. Please perform it while driving." + 2sec silence
            gt.loc[:, "actual_start_time"] = gt["startime"] + init_duration - 5.0
            gt.loc[:, "start_response"] = gt["actual_start_time"] + 2.5
            gt.loc[:, "end_response"] = gt["actual_start_time"] + 5.0
            gt.loc[:, "expected_response"] = gt["digit"]
        return gt

    def extract_non_silent_segments(self, monaural_audio: np.array) -> list:
        """
        This function extracts non-silent segments from the given sequence of monaural audio.
        The audio sequence is broken down to segments of STEP_SIZE.
        When the audio level is always lower than SILENT_THRESH, the segment is estimated as silent.

        Args:
            monaural_audio (np.array): A monaural audio sequence as 1D array.

        Returns:
            A list of pairs that contains index of start and end of non-silent segments.

        """

        STEP_SIZE_SEC = 0.1
        """
        An interval (seconds) to identify the silent segments.
        """
        STEP_SIZE = int(whisper.audio.SAMPLE_RATE * STEP_SIZE_SEC)
        """
        An interval (number of samples) to identify the silent segments.
        """
        SILENCE_THRESH = 0.05
        """
        A audio level threshold to determine if the section is silent.
        Currently, it's an empirical value specific to the Thunderhill training videos.
        TODO(hiro.yasuda): Determine the value based on the data.
        """

        # Check the audio level.
        is_silent = abs(monaural_audio) < SILENCE_THRESH
        is_silent_segment = [
            all(is_silent[i : i + STEP_SIZE]) for i in range(0, len(monaural_audio) - STEP_SIZE + 1, STEP_SIZE)
        ]

        # Extract start and end indices.
        segments = []
        consecutive_cnt = 0
        start_index = -1
        for i, silent in enumerate(is_silent_segment):
            index = i * STEP_SIZE + int(STEP_SIZE * 0.5)
            if not silent:
                if consecutive_cnt == 0:
                    start_index = index - int(STEP_SIZE * 0.5)
                consecutive_cnt += 1
            else:
                if consecutive_cnt > 0:
                    segments.append((start_index, index + int(STEP_SIZE * 0.5)))
                consecutive_cnt = 0

        # revert to seconds
        segments = [(s[0] / whisper.audio.SAMPLE_RATE, s[1] / whisper.audio.SAMPLE_RATE) for s in segments]
        return segments

    def proc_one(self, audio_file, hazard, task):
        results = []
        gt = self.load_gt(hazard, task)

        # Load the raw audio from the video.
        print("audio loaded", str(audio_file))
        audio_data = whisper.load_audio(str(audio_file))

        # De-noise the audio.
        denoised_audio = nr.reduce_noise(y=audio_data, sr=whisper.audio.SAMPLE_RATE, n_jobs=-1)

        # extract answering periods
        chunks = []
        misheard_chunks = []
        for i, row in gt.iterrows():
            expected_response = row["expected_response"]
            if task == "nback_task":
                expected_response = str(int(expected_response))
            start = int((row["start_response"] + self.offset_start) * whisper.audio.SAMPLE_RATE)
            end = int((row["end_response"] + self.offset_end) * whisper.audio.SAMPLE_RATE)

            if end > len(denoised_audio):
                break

            # if i > 5:
            #     break

            chunk = denoised_audio[start:end]

            # Extract non silent segments.
            original_non_silent_segments = self.extract_non_silent_segments(chunk)

            result_i = self.whisper_instance.model.transcribe(chunk, language="en")

            if self.whisper_instance.model_source == "whisper":
                ans = result_i["text"]
            elif self.whisper_instance.model_source == "stable_whisper":
                ans = result_i.segments
            else:
                ans = None

            # tmp_file = f"chunk_{i}.wav"
            # numpy2audiosegment(chunk).export(tmp_file, format="wav")

            print(
                i,
                row["start_response"],
                row["end_response"],
                f"gt:{expected_response}, ans:{ans}, nonsilent_seg:{original_non_silent_segments}",
            )

            results.append(
                {
                    "index": i,
                    "expected_res_start": row["start_response"],
                    "expected_res_end": row["end_response"],
                    "gt": expected_response,
                    "answer": ans,
                    "nonsilent_seg": original_non_silent_segments,
                    "whisper_result": result_i,
                }
            )

            if self.write_all_audio_chunks:
                chunks.append(chunk)

            if self.write_misheard_audio_chunks:
                norm_gt = normalize_text(expected_response)
                norm_ans = normalize_text(ans)
                is_match = is_included(norm_ans, norm_gt)
                is_nonsilent = nonsilent_valid(original_non_silent_segments)
                if not is_match and is_nonsilent:
                    misheard_chunks.append(
                        {
                            "audio": chunk,
                            "normalized_gt": norm_gt,
                            "normalized_ans": norm_ans,
                        }
                    )

        return results, chunks, misheard_chunks

    def proc_all(self):
        results = None
        misheard_index = 0
        for f in tqdm(self.audio_files):
            p = f.split("/")
            PID, PART, SEC, HAZARD, TASK = p[-7], p[-6], p[-5], p[-4], p[-3]
            print(PID, PART, SEC, HAZARD, TASK)
            if HAZARD != "practice" and TASK != "no_task":
                results, chunks, misheard_chunks = self.proc_one(f, HAZARD, TASK)

                # write result
                out_path = self.output_dir / PID / PART / SEC / HAZARD / TASK / Path("transcript.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=4)

                if self.write_all_audio_chunks:
                    for i, chunk in enumerate(chunks):
                        tmp_file = self.output_dir / PID / PART / SEC / HAZARD / TASK / Path(f"{i}.wav")
                        numpy2audiosegment(chunk).export(tmp_file, format="wav")

                if self.write_misheard_audio_chunks:
                    for chunk_dic in misheard_chunks:
                        tmp_file = (
                            Path(out_path).parent
                            / f"inproc_chunk_{misheard_index}_gt{chunk_dic['normalized_gt']}_ans{chunk_dic['normalized_ans'].replace('.','').replace(' ','')}.wav"
                        )
                        numpy2audiosegment(chunk_dic["audio"]).export(tmp_file, format="wav")
                        misheard_index += 1

        return results

    def set_mapper(self):
        mapper = {
            "1a-pedestrian_pop_out": {
                "nback_task": "nback/nback_every2.5_pttn1.csv",
                "statement_task": "statement/state_pttn1.csv",
            },
            "2-vehicle_pop_out": {
                "nback_task": "nback/nback_every2.5_pttn2.csv",
                "statement_task": "statement/state_pttn2.csv",
            },
            "2b-vehicle_door_open_hazard": {
                "nback_task": "nback/nback_every2.5_pttn3.csv",
                "statement_task": "statement/state_pttn3.csv",
            },
            "3c-pedestrian_pop_out": {
                "nback_task": "nback/nback_every2.5_pttn4.csv",
                "statement_task": "statement/state_pttn4.csv",
            },
            "5-vehicle_run_stop": {
                "nback_task": "nback/nback_every2.5_pttn5.csv",
                "statement_task": "statement/state_pttn5.csv",
            },
            "6e-vehicle_run_red_light": {
                "nback_task": "nback/nback_every2.5_pttn6.csv",
                "statement_task": "statement/state_pttn6.csv",
            },
            "7a-vehicle_run_red_light": {
                "nback_task": "nback/nback_every2.5_pttn7.csv",
                "statement_task": "statement/state_pttn7.csv",
            },
            "8a-pedestrian_pop_out": {
                "nback_task": "nback/nback_every2.5_pttn8.csv",
                "statement_task": "statement/state_pttn8.csv",
            },
        }
        return mapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="dir to store transcription and misheared audio segments.",
        default="/data/motion-simulator-logs/Processed/Features/transcript",
    )
    parser.add_argument("--path_filter", type=str, help="specify string to filter the mp3 path", default=None)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    ww = WhisperWrapper(model_source="whisper")
    cdp = CDParser(ww, output_dir=args.output_dir, path_filter=args.path_filter)
    results = cdp.proc_all()
