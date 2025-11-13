#!/usr/bin/env python3
import argparse
import os
import warnings
from multiprocessing import Pool

import pandas as pd

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.gaze_features import GazeFixationSaccade
from features.output_writer import OutputWriter

warnings.simplefilter("ignore")


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Get gaze fixation and saccade features
    tobii_df = sample.features["experiment_tobii_frame"].table
    fps = dataset.config.features["experiment_tobii_frame"]["fps"]
    gfs = GazeFixationSaccade()

    output_dfs = []
    for side in ["left", "right"]:
        fixation_saccade_df = gfs.process(tobii_df, 5.0, fps, 1, side)
        fixation_saccade_df = fixation_saccade_df.add_prefix(f"gaze_{side}_", axis="columns")
        output_dfs.append(fixation_saccade_df)

    output_dfs = pd.concat(output_dfs, axis="columns")

    # Save the gaze entropy features
    output_writer.save_tabular(output_dfs, "gaze_fixation_saccade")

    print(output_writer.out_sample_path)


def process_sample_wrapper(process_args):
    process_sample(**process_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="~/data/IDD/Processed",
        help="path to the top-level of the input resampled data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="~/data/IDD/Processed/Resampled_previous_60_derived",
        help="path to the top-level of the output derived data directory",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="The number of processes to use for resampling (0 just uses main thread)",
    )
    args = parser.parse_args()
    args.in_dir = os.path.expanduser(args.in_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    # Specify input features needed
    config = DBM_Dataset_Config()
    config.base_path = args.in_dir
    config.index_relative_path = "Resampled_previous_60"

    config.features = {
        "experiment_tobii_frame": {
            "relative_path": "Resampled_previous_60",
            "type": "tabular",
            "fps": 60,
            "columns": [
                "tobii_left_eye_gaze_pt_in_display_x",
                "tobii_right_eye_gaze_pt_in_display_x",
                "tobii_left_eye_gaze_pt_in_display_y",
                "tobii_right_eye_gaze_pt_in_display_y",
            ],
        },
    }

    config.scenario_name_filter = "driving/[0-9].*"
    config.chunk_strategy = "full"
    dataset = DBM_Dataset(config)

    # Gather participant process args
    all_process_args = []
    for sample_number in range(len(dataset)):
        process_args = {}
        process_args["dataset"] = dataset
        process_args["sample_number"] = sample_number
        process_args["base_out_dir"] = args.out_dir
        all_process_args.append(process_args)

    # Loop through samples and process
    if args.num_processes <= 0:
        for process_args in all_process_args:
            process_sample_wrapper(process_args)
    else:
        with Pool(args.num_processes) as pool:
            pool.map(process_sample_wrapper, all_process_args)


if __name__ == "__main__":
    main()
