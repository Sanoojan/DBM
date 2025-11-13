import os
import pickle

import cv2
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


class OutputWriter:
    def __init__(self, sample, base_out_dir):
        self.sample = sample
        self.out_sample_path = os.path.join(base_out_dir, self.sample.relative_path)
        os.makedirs(self.out_sample_path, exist_ok=True)

    def verify_single_index(self, csv_data, index_name, index_type, file_name):
        num_index = len(csv_data.index.names)
        if num_index != 1:
            raise Exception(f"Output dataframe for {file_name} must have single index, not be size {num_index}.")
        if csv_data.index.name != index_name:
            raise Exception(
                f"Output dataframe for {file_name} must have index named {index_name}, not {csv_data.index.name}."
            )
        if csv_data.index.dtype != index_type:
            raise Exception(
                f"Output dataframe for {file_name} must have index with type {index_type}, not {csv_data.index.dtype}."
            )

    def save_tabular(self, csv_data, file_name):
        # Verify that has single index named resampled_epoch_ns of type int64
        self.verify_single_index(csv_data, "resampled_epoch_ns", np.int64, file_name)

        # Verify table length matches expected output length
        target_length = self.sample.chunk.resample_times.shape[0]
        actual_length = csv_data.shape[0]
        if target_length != actual_length:
            raise Exception(
                f"Output dataframe for {file_name} must be {target_length} frames long, not {actual_length}."
            )

        # Sort data
        csv_data.sort_index(level=0, inplace=True)

        # Write data
        csv_data.to_csv(os.path.join(self.out_sample_path, f"{file_name}.csv"), index=True)

    def save_dictionary_array(self, data_dict, file_name):
        # Verify input is dictionary
        if not isinstance(data_dict, dict):
            raise Exception("Input data should be a dictionary.")

        # Verify input has time key
        if "resampled_epoch_ns" not in data_dict:
            raise Exception("Data must contain resampled_epoch_ns key.")
        num_samples = len(data_dict["resampled_epoch_ns"])

        # Verify that all inputs are numpy arrays and have first dimension of either 1 or num_samples
        for key, value in data_dict.items():
            if not isinstance(value, np.ndarray):
                raise Exception(f"Entry {key} is not a numpy array.")
            if value.shape[0] != 1 and value.shape[0] != num_samples:
                raise Exception(
                    f"Entry {key} has shape {value.shape} when first dimension should be 1 or {num_samples}."
                )

        # Write data
        file_path = os.path.join(self.out_sample_path, f"{file_name}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(data_dict, file)

    def save_object_attributes(self, csv_data, file_name):
        # Verify that has single index named object_id of type int64
        self.verify_single_index(csv_data, "object_id", np.int64, file_name)

        # Sort data
        csv_data.sort_index(level=0, inplace=True)

        # Write data
        csv_data.to_csv(os.path.join(self.out_sample_path, f"{file_name}.csv"), index=True)

    def save_object_tracks(self, object_dict, file_name):
        # Gather and verify the tables from each object
        all_tracks = []
        for object_id, object_tracks in object_dict.items():
            # Verify the ID and table
            if not isinstance(object_id, np.int64) and not isinstance(object_id, int):
                raise Exception(f"Object id {object_id} should be of np.int64 or int type, not {type(object_id)}")
            self.verify_single_index(object_tracks, "resampled_epoch_ns", np.int64, file_name)

            # Convert the table to have double index and append to list
            object_tracks.reset_index(inplace=True)
            object_tracks["object_id"] = object_id
            object_tracks.set_index(["resampled_epoch_ns", "object_id"], drop=True, inplace=True)
            all_tracks.append(object_tracks)
        all_tracks = pd.concat(all_tracks, axis=0)

        # Sort data
        all_tracks.sort_index(level=0, inplace=True)

        # Write data
        all_tracks.to_csv(os.path.join(self.out_sample_path, f"{file_name}.csv"), index=True)

    class VideoFrameWriter:
        def __init__(self, output_writer, file_name, extension, fourcc, fps, frame_size):
            self.output_writer = output_writer
            self.file_name = file_name
            self.frame_size = frame_size
            video_dir = os.path.join(output_writer.out_sample_path, file_name)
            os.makedirs(video_dir, exist_ok=True)
            self.video_path = os.path.join(video_dir, f"video{extension}")
            self.timing_path = os.path.join(video_dir, "frame_timing.csv")
            self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, (self.frame_size[1], self.frame_size[0]))
            self.frame_timing = []

        def write(self, time, image_data, timing_row={}):
            if image_data.shape[0] != self.frame_size[0] or image_data.shape[1] != self.frame_size[1]:
                raise Exception(
                    f"Output video frame size ({self.frame_size}) and written frame size ({image_data.shape}) don't match"
                )
            self.writer.write(image_data)
            timing_row["resampled_epoch_ns"] = time
            timing_row["resampled_frame"] = len(self.frame_timing)
            self.frame_timing.append(timing_row)

        def close(self):
            self.writer.release()
            csv_data = pd.DataFrame(self.frame_timing).set_index("resampled_epoch_ns")

            # Verify table length matches expected output length
            target_length = self.output_writer.sample.chunk.resample_times.shape[0]
            actual_length = csv_data.shape[0]
            if target_length != actual_length:
                raise Exception(
                    f"Output number of frames and timings for video {self.file_name} must be {target_length} frames long, not {actual_length}."
                )

            # Write data
            csv_data.to_csv(self.timing_path, index=True)

    def get_video_frame_writer(self, file_name, extension, fourcc, fps, frame_size):
        return self.VideoFrameWriter(self, file_name, extension, fourcc, fps, frame_size)
