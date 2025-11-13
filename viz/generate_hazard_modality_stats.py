#!/usr/bin/env python3
import cv2
import numpy as np
from numpy.dtypes import StringDType
from torch.utils.data import DataLoader

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config


def impaired_condition_to_text(condition):
    if condition is None:
        return "all"
    elif condition:
        return "yes"
    else:
        return "no"


class Logger:
    def __init__(self, out_path=None):
        if out_path is not None:
            self.out_file = open(out_path, "w")
        else:
            self.out_file = None

    def close(self):
        if self.out_file is not None:
            self.out_file.close()

    def print(self, message=""):
        print(message)
        if self.out_file is not None:
            self.out_file.write(message)


def get_stats_string(dataframe):
    num_scenarios = dataframe.shape[0]
    num_hours = (dataframe["duration_s"].sum() / 60.0) / 60.0
    return f",{num_scenarios} ({num_hours:.1f})"


def main():
    # Load the entire dataset
    config = DBM_Dataset_Config()
    config.index_relative_path = "Resampled_previous_10"

    config.features = {
        "tobii": {
            "relative_path": "Resampled_previous_10",
            "file_name": "experiment_tobii_frame",
            "fps": 10,
            "type": "tabular",
            "columns": [
                "tobii_left_eye_gaze_pt_validity",
            ],
        },
        "carla_actors": {
            "relative_path": "Resampled_previous_10",
            "file_name": "carla_actor_list",
            "type": "object_attributes",
            "object_id_column": "carla_actor_id",
            "column_types": {"carla_actor_type": StringDType},
            "max_objects": 32,
        },
        "ado_tracks": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "object_tracks",
            "fps": 10,
            "columns": [
                "carla_objects_pose_x",
                "carla_objects_pose_y",
                "carla_objects_pose_z",
            ],
            "max_objects": 32,
        },
        "cam_front": {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (960 // 2, 400 // 2, 3),
            "interpolation": cv2.INTER_AREA,
            "columns": [
                "resampled_frame",
            ],
            "skip_frames": True,
        },
    }

    config.chunk_duration = 1.0
    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
    config.scenario_name_filter = "driving/[0-9].*"

    dataset = DBM_Dataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate, num_workers=4)

    # num_has_tobii = 0
    # total_num = 0
    # duration_with_tobii = 0
    # for batch in dataloader:
    #     has_tobii = np.all(batch.features["tobii"].table_valid.squeeze(-1), axis=-1)
    #     num_has_tobii += np.sum(has_tobii)
    #     total_num += len(has_tobii)
    #     duration_with_tobii += np.sum(batch.scenario.duration_s[has_tobii])
    # print(num_has_tobii, total_num, num_has_tobii / total_num, duration_with_tobii / (60*60))

    num_has_tobii = 0
    total_num = 0
    duration_with_tobii = 0
    for batch in dataloader:
        has_tobii = batch.features["ado_tracks"].file_valid == 1
        num_has_tobii += np.sum(has_tobii)
        total_num += len(has_tobii)
        duration_with_tobii += np.sum(batch.scenario.duration_s[has_tobii])
    print(num_has_tobii, total_num, num_has_tobii / total_num, duration_with_tobii / (60 * 60))
    exit()

    # Setup logger
    logger = Logger()

    # Header for driving data
    logger.print("Driving Data")
    header_row = (
        "Intoxicated,Cog. Distracted",
        "All",
        "With Ego Tracks",
    )
    logger.print(header_row)

    # Each row filters over cognitive task and intoxication
    for intoxicated_filter in [False, True, None]:
        for cd_filter in [False, True, None]:
            # Form impairment subset
            impairment_subset = dataset.scenarios
            if intoxicated_filter is not None:
                impairment_subset = impairment_subset[impairment_subset["intoxicated"] == intoxicated_filter]
            if cd_filter is not None:
                impairment_subset = impairment_subset[(impairment_subset["cognitive_task"] != "no_task") == cd_filter]

            # First columns describe the impairment_subset conditions
            row_string = f"{impaired_condition_to_text(intoxicated_filter)},{impaired_condition_to_text(cd_filter)}"

            # Stats for all data
            row_string += get_stats_string(impairment_subset)

            # Require ego tracks to exist
            dataset.config.sources[0]["include_carla_objects"] = True

            # Print row
            logger.print(row_string)

    # Close logger
    logger.close()


if __name__ == "__main__":
    main()
