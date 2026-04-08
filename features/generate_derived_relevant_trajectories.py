#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.output_writer import OutputWriter
from utils.carla import get_carla_object_yaw


def process_sample(dataset, sample_number, base_out_dir):
    sample = dataset[sample_number]
    output_writer = OutputWriter(sample, base_out_dir)

    # Check that the required input features exist and read
    if "carla_actors" not in sample.features or "carla_tracks" not in sample.features:
        return
    carla_actors = sample.features["carla_actors"].attributes
    carla_tracks = sample.features["carla_tracks"].tracks

    # Determine the ego - needed to separate
    ego_candidates = carla_actors[carla_actors["carla_actor_rolename"] == "hero"].index.values
    if len(ego_candidates) != 1:
        return
    ego_id = ego_candidates[0]

    # Separate the ego tracks
    ego_tracks = carla_tracks[ego_id]

    # Calculate relevance for each track
    coord_names = ["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]
    for object_id, object_tracks in carla_tracks.items():
        overlap_start = max(object_tracks.index[0], ego_tracks.index[0])
        overlap_end = min(object_tracks.index[-1], ego_tracks.index[-1])
        delta_pos = (
            object_tracks.loc[overlap_start:overlap_end, coord_names]
            - ego_tracks.loc[overlap_start:overlap_end, coord_names]
        )
        object_tracks.loc[overlap_start:overlap_end, "relevance"] = np.sqrt(
            np.sum(np.square(delta_pos.to_numpy()), axis=-1)
        )

    # Calculate features for each object
    for object_id in carla_tracks.keys():
        object_tracks = carla_tracks[object_id]
        # new_features = []
        # for current_index in object_tracks.index:
        #     new_row = {}
        #     yaw_rad = get_carla_object_yaw(object_tracks, current_index)
        #     new_row["yaw"] = np.rad2deg(yaw_rad)
        #     new_features.append(new_row)
        # new_features = pd.DataFrame(new_features, index=object_tracks.index)
        # carla_tracks[object_id] = pd.concat([carla_tracks[object_id], new_features], axis="columns")
        object_tracks["yaw"] = np.rad2deg(get_carla_object_yaw(object_tracks))

    # Output the trajectory features
    output_writer.save_object_tracks(carla_tracks, "relevant_trajectories")

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
        "carla_actors": {
            "relative_path": "Resampled_previous_10",
            "file_name": "carla_actor_list",
            "type": "object_attributes",
            "object_id_column": "carla_actor_id",
        },
        "carla_tracks": {
            "relative_path": "Resampled_previous_10",
            "file_name": "carla_objects",
            "type": "object_tracks",
            "fps": 10,
            "object_id_column": "carla_objects_id",
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
