#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.output_writer import OutputWriter
from features.shrink_video import shrink_video


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    shrink_video(sample, output_writer)

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
        "cam_front": {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (400 // 4, 960 // 4, 3),
            "skip_frames": True,
        }
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
