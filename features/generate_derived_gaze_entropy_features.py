#!/usr/bin/env python3
import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.gaze_features import GazeEntropyProcess
from features.output_writer import OutputWriter

warnings.simplefilter("ignore")


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Get gaze entropy
    tobii_df = sample.features["experiment_tobii_frame"].table
    fps = dataset.config.features["experiment_tobii_frame"]["fps"]
    entropy_df = pd.DataFrame([], index=tobii_df.index)

    for side in ["left", "right"]:
        gaze_x = tobii_df[f"tobii_{side}_eye_gaze_pt_in_display_x"].to_numpy()
        gaze_y = tobii_df[f"tobii_{side}_eye_gaze_pt_in_display_y"].to_numpy()

        entropy_process = GazeEntropyProcess(ws=5.0, grid_x=5, grid_y=1, stride=1, fps=fps)
        indexes, entropy_transition, entropy_stationary = entropy_process.calc_ent_rolling(gaze_x, gaze_y)

        # Convert into a dataframe
        entropy_df.loc[tobii_df.index[indexes], f"gaze_{side}_transition_entropy"] = entropy_transition
        entropy_df.loc[tobii_df.index[indexes], f"gaze_{side}_stationary_entropy"] = entropy_stationary

    # Save the gaze entropy features
    output_writer.save_tabular(entropy_df, "gaze_entropy")

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
