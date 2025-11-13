import numpy as np


def get_feature_set_config(feature_set, normalize_temporal="none", normalize_aggregate="none", include_objects=False):
    if feature_set == "main":
        features = {
            "ego_tracks": {
                "relative_path": "Resampled_previous_10_derived",
                "type": "tabular",
                "fps": 10,
                "columns": [
                    "carla_objects_pose_x",
                    "carla_objects_pose_y",
                ],
                "normalize": normalize_temporal,
                "aggregate": [
                    {"column": "longitudinal_speed", "function": "pos_mean"},
                    {"column": "longitudinal_acceleration", "function": "pos_mean"},
                    {"column": "longitudinal_acceleration", "function": "neg_mean"},
                ],
                "normalize_aggregate": normalize_aggregate,
            },
            "steering_reversals": {
                "relative_path": "Resampled_previous_10_derived",
                "type": "tabular",
                "fps": 10,
                "aggregate": [
                    {"column": "reversals_0_5", "function": "sum_rate"},
                    {"column": "reversals_2_5", "function": "sum_rate"},
                ],
                "normalize_aggregate": normalize_aggregate,
            },
            "experiment_tobii_frame": {
                "relative_path": "Resampled_previous_60",
                "type": "tabular",
                "fps": 60,
                "normalize": normalize_temporal,
                "aggregate": [
                    {
                        "name": "tobii_eye_pupil_diameter",
                        "columns": ["tobii_left_eye_pupil_diameter", "tobii_right_eye_pupil_diameter"],
                        "function": "mean",
                    },
                ],
                "normalize_aggregate": normalize_aggregate,
            },
            "gaze_pitch_yaw": {
                "relative_path": "Resampled_previous_60_derived",
                "type": "tabular",
                "fps": 60,
                "normalize": normalize_temporal,
                "aggregate": [
                    {"name": "gaze_pitch", "columns": ["gaze_left_pitch", "gaze_right_pitch"], "function": "std"},
                    {"name": "gaze_yaw", "columns": ["gaze_left_yaw", "gaze_right_yaw"], "function": "std"},
                ],
                "normalize_aggregate": normalize_aggregate,
            },
            "gaze_fixation_saccade": {
                "relative_path": "Resampled_previous_60_derived",
                "type": "tabular",
                "fps": 60,
                "aggregate": [
                    {
                        "name": "gaze_saccade",
                        "columns": ["gaze_left_saccade", "gaze_right_saccade"],
                        "function": "change_rate",
                    },
                    {
                        "name": "gaze_fixation",
                        "columns": ["gaze_left_fixation", "gaze_right_fixation"],
                        "function": "change_rate",
                    },
                ],
                "normalize_aggregate": normalize_aggregate,
            },
        }
    else:
        raise ValueError(f"Invalid features argument: {feature_set}")
    if include_objects:
        print("Including object data in the dataset")
        features["object_tracks"] = {
            "relative_path": "Resampled_previous_10",
            "file_name": "carla_objects",
            "type": "object_tracks",
            "max_objects": 48,
            "fps": 10,
            "columns": [
                "carla_objects_log_time",
                "carla_objects_pose_x",
                "carla_objects_pose_y",
                "carla_objects_pose_z",
                "carla_objects_pose_orientation_x",
                "carla_objects_pose_orientation_y",
                "carla_objects_pose_orientation_z",
                "carla_objects_pose_orientation_w",
                "carla_objects_twist_linear_vx",
                "carla_objects_twist_linear_vy",
                "carla_objects_twist_linear_vz",
                "carla_objects_bbox_x",
                "carla_objects_bbox_y",
                "carla_objects_bbox_z",
                "resampled_log_time",
                "smoothed_log_time",
            ],
            "object_id_column": "carla_objects_id",
        }

    return features
