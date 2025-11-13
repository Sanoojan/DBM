#!/usr/bin/env python3
"""
Script to test basic functionality of the ddd_2024 dataset
Developed to support IDD.
"""
import numpy as np

from ddd_2024_dataset import DDD2024Dataset, DDD2024DatasetConfig


def main():
    config = DDD2024DatasetConfig()
    config.base_path = "~/data/IDD/Processed/"
    config.index_relative_path = "Resampled_previous_10" 
    
    config.features = {
        "tobii": {
            "relative_path": "Resampled_previous_10",
            "file_name": "experiment_tobii_frame",
            "type": "tabular",
            "fps": 10,
            "columns": ["tobii_left_eye_gaze_pt_validity",]
        },
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
            "columns": ["carla_objects_pose_x"]
        },
        "cam_front": {
            "relative_path": "Resampled_previous_10",
            "file_name": "cam_front",
            "type": "video",
            "fps": 10,
            "resolution": (400//2, 960//2, 3),
        }
    }

    config.scenario_name_filter = ".*driving.*"
    
    config.chunk_strategy = "random"
    config.chunk_duration = 10.
    config.chunks_per_scenario = 2
    config.cache_chunks = True
    
    dataset = DDD2024Dataset(config)
    print("Number scenarios:", len(dataset))
    for i in range(5):
        sample = dataset[i]
        print(sample)
    print("Repeat")
    for i in range(2):
        sample = dataset[i]
        print(sample)
        
if __name__ == "__main__":
    main()
