#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import signal

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.output_writer import OutputWriter


def find_upward_reversals(wheel_smoothed, stationary_indices, min_theta):
    prev_stationary = wheel_smoothed[stationary_indices[0]]
    reversals = []
    for i in range(1, len(stationary_indices)):
        current_stationary = wheel_smoothed[stationary_indices[i]]
        if current_stationary - prev_stationary >= min_theta:
            reversals.append(stationary_indices[i])
            prev_stationary = current_stationary
        elif current_stationary < prev_stationary:
            prev_stationary = current_stationary
    return reversals


def find_reversals(wheel_smoothed, stationary_indices, min_theta):
    upward = find_upward_reversals(wheel_smoothed, stationary_indices, min_theta)
    downward = find_upward_reversals(-wheel_smoothed, stationary_indices, min_theta)
    reversals = upward + downward
    return sorted(set(reversals))


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Determine steering values
    steering_df = sample.features["telemetry_roadwheel_angle"].table
    wheel_angle = steering_df["telemetry_roadwheel_angle_data"].to_numpy()

    # Low pass filter
    fps = dataset.config.features["telemetry_roadwheel_angle"]["fps"]
    b, a = signal.butter(N=2, Wn=2.0, btype="low", fs=fps)
    wheel_smoothed = signal.filtfilt(b, a, wheel_angle)

    # Detect stationary points
    delta_wheel = np.diff(wheel_smoothed, prepend=wheel_smoothed[0])
    is_zero = delta_wheel == 0.0
    is_zero[0] = False
    sign_wheel = np.sign(delta_wheel).astype(int)
    diff_sign = sign_wheel[:-1] - sign_wheel[1:]
    is_two_diff = diff_sign == 2
    is_two_diff = np.concat(
        [
            is_two_diff,
            [
                False,
            ],
        ]
    )
    is_stationary = np.logical_or(is_zero, is_two_diff)
    stationary_indices = np.where(is_stationary)[0]

    # Find reversal points
    reversal_stats = {}
    min_thetas = [0.5, 2.5]
    for min_theta in min_thetas:
        reversals = find_reversals(wheel_smoothed, stationary_indices, min_theta)
        reversal_arr = np.zeros((len(wheel_smoothed)), dtype=int)
        reversal_arr[reversals] = 1
        reversal_stats[f"reversals_{min_theta}".replace(".", "_")] = reversal_arr
    reversal_df = pd.DataFrame(reversal_stats, index=steering_df.index)

    # Save the reversals
    output_writer.save_tabular(reversal_df, "steering_reversals")

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
        "telemetry_roadwheel_angle": {
            "relative_path": "Resampled_previous_10",
            "type": "tabular",
            "fps": 10,
            "columns": [
                "telemetry_roadwheel_angle_data",
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
