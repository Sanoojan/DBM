#!/usr/bin/env python3
import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.output_writer import OutputWriter

warnings.simplefilter("ignore")


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Check that video exists
    if "cam_depth" not in sample.features:
        return

    # Get gaze data
    tobii_df = sample.features["experiment_tobii_frame"].table

    # Prepare output
    depth_df = pd.DataFrame([], index=tobii_df.index)
    depth_df["gaze_left_depth"] = np.nan
    depth_df["gaze_right_depth"] = np.nan

    # loop through depth frames
    frame_reader = sample.features["cam_depth"].get_frame_reader()
    for frame in frame_reader.read_frames():
        # Get tobii data
        if frame.time not in tobii_df.index:
            continue
        tobii_row = tobii_df.loc[frame.time]

        for side in ["left", "right"]:
            gaze_x = tobii_row[f"tobii_{side}_eye_gaze_pt_in_display_x"]
            gaze_y = tobii_row[f"tobii_{side}_eye_gaze_pt_in_display_y"]

            # Convert to screen pixel index
            if np.isnan(gaze_x) or np.isnan(gaze_y):
                continue
            gaze_x = int(round(gaze_x * frame.image_data.shape[1]))
            gaze_y = int(round(gaze_y * frame.image_data.shape[0]))
            if gaze_x < 0 or gaze_x >= frame.image_data.shape[1] or gaze_y < 0 or gaze_y >= frame.image_data.shape[0]:
                continue

            # Get depth at index
            depth_df.loc[frame.time, f"gaze_{side}_depth"] = frame.image_data[gaze_y, gaze_x]

    # Save the gaze depth features
    output_writer.save_tabular(depth_df, "gaze_depth")
    frame_reader.close()

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
        default="~/data/IDD/Processed/Resampled_previous_10_derived",
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
    config.index_relative_path = "Resampled_previous_10"

    config.features = {
        "experiment_tobii_frame": {
            "relative_path": "Resampled_previous_10",
            "type": "tabular",
            "fps": 10,
            "columns": [
                "tobii_left_eye_gaze_pt_in_display_x",
                "tobii_right_eye_gaze_pt_in_display_x",
                "tobii_left_eye_gaze_pt_in_display_y",
                "tobii_right_eye_gaze_pt_in_display_y",
            ],
        },
        "cam_depth": {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "video_type": "depth",
            "resolution": (400, 960, 1),
            "skip_frames": True,
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
