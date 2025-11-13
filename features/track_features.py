import os

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from utils.carla import get_carla_object_yaw
from utils.math import clamp_angle_deg


def split_ego_ado_tracks(sample, output_writer):
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

    # Add additional derived features to the ego tracks
    ego_derived_rows = []
    for current_ns in ego_tracks.index:
        ego_derived_row = {}
        start_ns = current_ns - 2500000000
        end_ns = current_ns + 2500000000
        ego_subset = ego_tracks.loc[start_ns:end_ns]

        # Determine ego yaw
        ego_yaw_rad = get_carla_object_yaw(ego_subset, current_ns)
        ego_derived_row["yaw"] = np.rad2deg(ego_yaw_rad)

        # Normalize ego tracks based on yaw so that positive y is logitudinal axis
        ego_xy = ego_subset[["carla_objects_pose_x", "carla_objects_pose_y"]].to_numpy()
        ego_xy -= ego_xy[0]
        c, s = np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)
        rotate_R = np.array(((c, -s), (s, c)))
        ego_xy = ego_xy @ rotate_R

        # Fit a spline and predict its derivatives (lateral and logitudinal speed and acceleration)
        times = (ego_subset.index - current_ns) / 1e9
        weights = np.cos((times / 2.5) * (np.pi / 2.0))  # Hann window
        tck, _ = interpolate.splprep([ego_xy[:, 0], ego_xy[:, 1]], w=weights, u=times)
        speeds = interpolate.splev(0, tck, der=1)
        ego_derived_row["lateral_speed"] = speeds[0]
        ego_derived_row["longitudinal_speed"] = speeds[1]
        accelerations = interpolate.splev(0, tck, der=2)
        ego_derived_row["lateral_acceleration"] = accelerations[0]
        ego_derived_row["longitudinal_acceleration"] = accelerations[1]
        jerks = interpolate.splev(0, tck, der=3)
        ego_derived_row["lateral_jerk"] = jerks[0]
        ego_derived_row["longitudinal_jerk"] = jerks[1]
        ego_derived_rows.append(ego_derived_row)
    ego_derived_df = pd.DataFrame(ego_derived_rows, index=ego_tracks.index)
    ego_tracks = ego_tracks.join(ego_derived_df)

    # Save the ego tracks with added derived data
    output_writer.save_tabular(ego_tracks, "ego_tracks")

    # Remove the ego from the tracks
    del carla_tracks[ego_id]

    # Determine if the ado tracks are valid
    if len(carla_tracks) < 10:
        return

    # Output the ado tracks
    output_writer.save_object_tracks(carla_tracks, "ado_tracks")

    # Determine the hazard - all pedestrians, any vehicles alive the entire scenario
    ego_start_ns = ego_tracks.index[0]
    ego_end_ns = ego_tracks.index[-1]
    hazard_id = -1
    min_hazard_score = np.inf
    for ado_id, ado_tracks in carla_tracks.items():
        # Determine ego distance to potential hazard
        ado_ns = ado_tracks.index.get_level_values("resampled_epoch_ns")
        ego_subset = ego_tracks.loc[ado_ns]
        ego_xyz = ego_subset[["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]].to_numpy()
        ado_xyz = ado_tracks[["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]].to_numpy()
        distances = np.sqrt(np.sum(np.square(ego_xyz - ado_xyz), axis=-1))

        # All pedestrians are definitely hazards
        ado_attributes = carla_actors.loc[ado_id]
        if ado_attributes.carla_actor_rolename == "pedestrian":
            hazard_id = ado_id
            break

        # Ego should get eventually close to the hazard
        hazard_score = (
            np.abs(ado_attributes.start_epoch_ns - ego_start_ns)
            + np.abs(ado_attributes.end_epoch_ns - ego_end_ns)
            + np.min(distances)
        )

        # All vehicle hazards should be a mercedes
        if ado_attributes.carla_actor_type != "vehicle.mercedes.coupe_2020":
            hazard_score += 10000

        # Determine if best hazard candidate (lowest score)
        if hazard_score < min_hazard_score:
            min_hazard_score = hazard_score
            hazard_id = ado_id

    # Append the hazard distance to the other ego track info
    hazard_ns = carla_tracks[hazard_id].index.get_level_values("resampled_epoch_ns")

    # Output the hazard tracks
    hazard_tracks = carla_tracks[hazard_id].droplevel("object_id")
    output_writer.save_tabular(hazard_tracks, "hazard_tracks")

    # Loop through each track and determine removal cause
    all_removal_stats = []
    for ado_id, ado_tracks in carla_tracks.items():
        # Calculate other deletion stats
        removal_stats = {}
        removal_stats["object_id"] = ado_id
        ado_tracks = ado_tracks.droplevel("object_id")
        ado_ns = ado_tracks.index
        removal_stats["time_ns"] = ado_ns[-1]
        ego_subset = ego_tracks.loc[ado_ns]
        rotate_yaw = ego_subset.iloc[-1]["yaw"]
        ego_xyz = ego_subset[["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]].values[-1]
        ado_xyz = ado_tracks[["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]].values[-1]
        ego_delta = ado_xyz - ego_xyz
        final_ego_distance = np.sqrt(np.sum(np.square(ego_delta)))
        removal_stats["distance_from_ego"] = final_ego_distance
        ego_direction_deg = np.rad2deg(np.arctan2(ego_delta[1], ego_delta[0]))
        ego_direction_deg = clamp_angle_deg(ego_direction_deg - rotate_yaw)
        removal_stats["direction_from_ego_deg"] = ego_direction_deg
        removal_stats["on_screen"] = (
            final_ego_distance < 100.0 and ego_direction_deg >= 45.0 and ego_direction_deg <= 135.0
        )

        # Check for ado that lasts until the end of the scenario
        if ado_ns[-1] == ego_end_ns:
            removal_stats["cause"] = "end"
            all_removal_stats.append(removal_stats)
            continue

        # Check for ado that gets too far from ego
        # Using 250 since may be delay before deletion
        if final_ego_distance >= 250.0:
            removal_stats["cause"] = "far"
            all_removal_stats.append(removal_stats)
            continue

        # Check for ado that stays still for too long (no more than 0.25 in 7 seconds)
        # Using 6.5 second since there may be a delay before deletion
        prev_ns = ado_ns[-1] - 7000000000
        prev_distance = -1
        if prev_ns in ado_tracks.index:
            prev_xyz = ado_tracks.loc[prev_ns][
                ["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]
            ].to_numpy()
            prev_distance = np.sqrt(np.sum(np.square(prev_xyz - ado_xyz)))
            if prev_distance <= 0.5:
                removal_stats["cause"] = "still"
                all_removal_stats.append(removal_stats)
                continue

        # Check for deletion by the hazard when within 100 meters (3c and 5)
        # Using 110 meters since there may be a delay before deletion
        final_hazard_distance = -1
        if (
            sample.scenario.scenario_name == "driving/3c-pedestrian_pop_out"
            or sample.scenario.scenario_name == "driving/5-vehicle_run_stop"
        ):
            hazard_subset = hazard_tracks.loc[ado_ns]
            hazard_xyz = hazard_subset[["carla_objects_pose_x", "carla_objects_pose_y", "carla_objects_pose_z"]].values[
                -1:
            ]
            final_hazard_distance = np.sqrt(np.sum(np.square(hazard_xyz - ado_xyz)))
            if final_hazard_distance <= 110.0:
                removal_stats["cause"] = "hazard"
                all_removal_stats.append(removal_stats)
                continue

        # Any other cause is unknown
        if ado_ns[0] == 0:
            removal_stats["cause"] = "previous"
        else:
            removal_stats["cause"] = "unknown"
        all_removal_stats.append(removal_stats)

    # Output the removal causes
    all_removal_stats = pd.DataFrame(all_removal_stats).set_index("object_id")
    output_writer.save_object_attributes(all_removal_stats, "ado_removal_stats")
