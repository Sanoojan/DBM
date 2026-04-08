#!/usr/bin/env python3
import argparse
import os
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.output_writer import OutputWriter

warnings.simplefilter("ignore")

# Semantic segmentation classes for reference
# 0: (0, 0, 142),  # unlabeled (in practice, some parts of kerb)
# 1: (128, 64, 128),  # road
# 2: (244, 35, 232),  # sidewalk
# 3: (70, 70, 70),  # building
# 4: (102, 102, 156),  # wall
# 5: (190, 153, 153),  # fence
# 6: (153, 153, 153),  # pole
# 7: (250, 170, 30),  # traffic light
# 8: (220, 220, 0),  # traffic sign
# 9: (107, 142, 35),  # vegetation
# 10: (152, 251, 152),  # terrain
# 11: (70, 130, 180),  # sky
# 12: (220, 20, 60),  # pedestrian
# 13: (255, 0, 0),  # rider
# 14: (0, 0, 142),  # car
# 15: (0, 0, 70),  # truck
# 16: (0, 60, 100),  # bus
# 17: (0, 80, 100),  # train
# 18: (0, 0, 230),  # motorcycle
# 19: (119, 11, 32),  # bicycle
# 20: (110, 190, 160),  # static immovable elements - bin/bollards/signs
# 21: (170, 120, 50),  # dynamic elements - movable trash bins, buggies
# 22: (55, 90, 80),  # other - everything that does not belong to any other category
# 23: (45, 60, 150),  # water
# 24: (157, 234, 50),  # road lines / markings
# 25: (81, 0, 81),  # other horizontal ground level structures, e.g. roundabouts
# 26: (150, 100, 100),  # Bridge
# 27: (230, 150, 140),  # RailTrack
# 28: (180, 165, 180),  # GuardRail


def reverse_argmax(x, axis):
    return x.shape[axis] - np.argmax(x[::-1], axis=axis) - 1


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Check that video exists
    if "cam_seg" not in sample.features:
        return

    # Get gaze data
    tobii_df = sample.features["experiment_tobii_frame"].table

    # Prepare output
    seg_df = pd.DataFrame([], index=tobii_df.index)
    seg_df["gaze_left_semantics"] = -1
    seg_df["gaze_right_semantics"] = -1

    # loop through semantic segmentation frames
    frame_reader = sample.features["cam_seg"].get_frame_reader()
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
            if (
                np.isnan(gaze_x)
                or np.isnan(gaze_y)
                or gaze_x < 0
                or gaze_x >= frame.image_data.shape[1]
                or gaze_y < 0
                or gaze_y >= frame.image_data.shape[0]
            ):
                continue

            # Check for NaN at location
            if np.isnan(frame.image_data[gaze_y, gaze_x]):
                continue

            # Get semantics at index
            seg_df.loc[frame.time, f"gaze_{side}_semantics"] = frame.image_data[gaze_y, gaze_x]

    # Save the gaze semantics features
    output_writer.save_tabular(seg_df, "gaze_semantics")
    frame_reader.close()

    print(output_writer.out_sample_path)


def process_sample_wrapper(process_args):
    process_sample(**process_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="dataset/Vehicle/No-Video",
        help="path to the top-level of the input resampled data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset/Vehicle/No-Video/Resampled_previous_10_derived",
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
        "cam_seg": {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "video_type": "segmentation",
            "resolution": (400, 960, 1),
            "skip_frames": True,
        },
    }

    config.scenario_name_filter = "driving/[0-9].*"
    config.chunk_strategy = "full"
    dataset = DBM_Dataset(config)
    print(len(dataset))

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
        for process_args in all_process_args[1:]:
            process_sample_wrapper(process_args)
    else:
        with Pool(args.num_processes) as pool:
            pool.map(process_sample_wrapper, all_process_args)


if __name__ == "__main__":
    main()
