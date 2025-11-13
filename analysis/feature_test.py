#!/usr/bin/env python3
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from features.feature_sets import get_feature_set_config


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
        default="~/data/IDD/Analysis/FeatureAnalysis",
        help="path to the top-level of the output derived data directory",
    )
    parser.add_argument(
        "--recache",
        action="store_true",
        help="whether or not to recache features for analysis",
    )
    args = parser.parse_args()
    args.in_dir = os.path.expanduser(args.in_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    # Specify input features needed
    config = DBM_Dataset_Config()
    config.base_path = args.in_dir
    config.index_relative_path = "Resampled_previous_10"

    config.features = get_feature_set_config("main", "none", "none")

    config.scenario_name_filter = "driving/[0-9].*"
    config.experiment_version = None
    # config.exclude_cd = ["statement_task"]
    # config.exclude_intoxicated = True

    # Use 90 second chunks, offset 30 seconds from the end to exclude hazard
    config.chunk_strategy = "end"
    config.chunk_end_offset = 30.0
    config.chunk_duration = 90.0
    config.chunks_per_scenario = 1

    # Load all data
    dataset = DBM_Dataset(config)

    # Loop through and calculate statistics
    os.makedirs(args.out_dir, exist_ok=True)
    chunk_stats_path = os.path.join(args.out_dir, "chunk_stats.pkl")
    if not os.path.isfile(chunk_stats_path) or args.recache:
        chunk_stats = []
        for sample_on, sample in enumerate(dataset):
            # Scenario metadata
            scenario_number = sample.scenario.scenario_name.split("/")[1].split("-")[0]
            participant_name = sample.participant.name
            sample_row = {}
            sample_row["scenario"] = scenario_number
            sample_row["participant"] = participant_name
            sample_row["experiment_version"] = sample.scenario.experiment_version
            sample_row["duration_s"] = sample.scenario.duration_s
            sample_row["round2"] = sample.scenario.round == "R2"
            sample_row["intoxicated"] = sample.scenario.intoxicated
            sample_row["cd"] = sample.scenario.cognitive_task != "no_task"

            # Crashes
            sample_row["crashes_total"] = sample.scenario.number_all_crashes
            sample_row["crashes_hazard"] = sample.scenario.number_hazard_crashes
            if np.isnan(sample_row["crashes_hazard"]):
                sample_row["crashes_passive"] = sample_row["crashes_total"]
            else:
                sample_row["crashes_passive"] = sample_row["crashes_total"] - sample_row["crashes_hazard"]

            # Gather other features
            sample_row.update(sample.features["ego_tracks"].aggregate.to_dict("records")[0])
            sample_row.update(sample.features["steering_reversals"].aggregate.to_dict("records")[0])
            sample_row.update(sample.features["gaze_pitch_yaw"].aggregate.to_dict("records")[0])
            sample_row.update(sample.features["gaze_fixation_saccade"].aggregate.to_dict("records")[0])
            sample_row.update(sample.features["experiment_tobii_frame"].aggregate.to_dict("records")[0])

            # Append to stats dataframe
            chunk_stats.append(sample_row)
            print(sample_on, "/", len(dataset))

        # Create chunk dataframe
        chunk_stats = pd.DataFrame(chunk_stats)

        # Cache features
        chunk_stats.to_pickle(chunk_stats_path)
    else:
        chunk_stats = pd.read_pickle(chunk_stats_path)

    # Test intoxication
    results = {}
    v1_subset = chunk_stats[chunk_stats["experiment_version"] == 1]
    intoxicated_subset = v1_subset[v1_subset["cd"] == False]
    results["intoxicated"] = stat_test(
        intoxicated_subset[intoxicated_subset["intoxicated"] == False],
        intoxicated_subset[intoxicated_subset["intoxicated"] == True],
        "intoxicated",
        args.out_dir,
    )

    # Test CD
    cd_subset = chunk_stats[chunk_stats["intoxicated"] == False]
    results["cd"] = stat_test(
        cd_subset[cd_subset["cd"] == False], cd_subset[cd_subset["cd"] == True], "cd", args.out_dir
    )

    # Test intoxication
    # v1_subset = chunk_stats[chunk_stats["experiment_version"] == 1]
    # intoxicated_subset = v1_subset[v1_subset["cd"] == True]
    # results["intoxicated_with_cd"] = stat_test(
    #     intoxicated_subset[intoxicated_subset["intoxicated"] == False],
    #     intoxicated_subset[intoxicated_subset["intoxicated"] == True],
    #     "intoxicated_with_cd",
    #     args.out_dir,
    # )

    # Test joint difference
    joint_neither_subset = v1_subset[np.logical_not(np.logical_xor(v1_subset["intoxicated"], v1_subset["cd"]))]
    results["joint"] = stat_test(
        joint_neither_subset[joint_neither_subset["intoxicated"] == False],
        joint_neither_subset[joint_neither_subset["intoxicated"] == True],
        "joint",
        args.out_dir,
    )

    # Test driver state difference
    # xor_subset = v1_subset[np.logical_xor(v1_subset["intoxicated"], v1_subset["cd"])]
    # results["intoxicated_vs_cd"] = stat_test(xor_subset[xor_subset["intoxicated"] == False], xor_subset[xor_subset["intoxicated"] == True], "intoxicated_vs_cd", args.out_dir)

    # Output results table
    test_columns = chunk_stats.columns.values
    start_idx = np.where(test_columns == "cd")[0][0] + 1
    test_columns = test_columns[start_idx:]
    for column in test_columns:
        if "crashes" in column:
            continue
        column_name = name_lookup[column] if column in name_lookup else column
        print(f"{column_name} ", end="")
        for condition, condition_results in results.items():
            mean_diff = condition_results[column]["mean_diff"] if column in condition_results else np.nan
            effect_size = condition_results[column]["effect"] if column in condition_results else np.nan
            pvalue = condition_results[column]["pvalue"] if column in condition_results else np.nan
            if pvalue >= 0.05:
                print(
                    "& \\textcolor{not-sig}{"
                    + f"{mean_diff:.3f}"
                    + "} & \\textcolor{not-sig}{"
                    + f"{effect_size:.3f}"
                    + "} & \\textcolor{not-sig}{"
                    + f"{pvalue:.4f}"
                    + "} ",
                    end="",
                )
            else:
                print(f"& {mean_diff:.3f} & {effect_size:.3f} & {pvalue:.4f} ", end="")
        print("\\\\")


name_lookup = {
    "longitudinal_speed__pos_mean": "Longitudinal Speed (m/s)",
    "longitudinal_acceleration__pos_mean": "Longitudinal Acceleration (m/s\\textsuperscript{2})",
    "longitudinal_acceleration__neg_mean": "Braking Acceleration (m/s\\textsuperscript{2})",
    "reversals_0_5__sum_rate": "Steering Reversals at 0.5 Degrees (\#/min)",
    "reversals_2_5__sum_rate": "Steering Reversals at 2.5 Degrees (\#/min)",
    "gaze_pitch__std": "Gaze Pitch Standard Dev. (Degrees)",
    "gaze_yaw__std": "Gaze Yaw Standard Dev. (Degrees)",
    "gaze_saccade__change_rate": "Saccades (\#/min)",
    "gaze_fixation__change_rate": "Fixations (\#/min)",
    "tobii_eye_pupil_diameter__mean": "Pupil Diameter (mm)",
}


def stat_test(baseline_df, condition_df, condition, out_dir):
    # Get test columns
    test_columns = baseline_df.columns.values
    start_idx = np.where(test_columns == "cd")[0][0] + 1
    test_columns = test_columns[start_idx:]

    # Setup output directory
    condition_out_dir = os.path.join(out_dir, condition)
    if os.path.isdir(condition_out_dir):
        shutil.rmtree(condition_out_dir)
    os.makedirs(condition_out_dir)

    # Paired test
    results = {}
    for test_column in test_columns:
        # Get data
        baseline_data_by_chunk = baseline_df[test_column].to_numpy()
        impaired_data_by_chunk = condition_df[test_column].to_numpy()
        baseline_data = baseline_df.groupby("participant", sort=True)[test_column].mean().to_numpy()
        impaired_data = condition_df.groupby("participant", sort=True)[test_column].mean().to_numpy()

        if len(baseline_data) != len(impaired_data):
            continue
        delta_data = impaired_data - baseline_data
        mean_diff = np.mean(delta_data)

        # Determine significance
        sig_test = stats.wilcoxon(delta_data, method="approx")
        pvalue = sig_test.pvalue
        # pvalue = stats.ttest_rel(baseline_data, impaired_data).pvalue
        # pvalue = stats.ttest_ind(baseline_data, impaired_data).pvalue
        results[test_column] = {
            "mean_diff": mean_diff,
            "pvalue": pvalue,
            "effect": np.abs(sig_test.zstatistic) / np.sqrt(baseline_data.shape[0]),
        }

        # Plot distribution of differences
        # plt.hist(baseline_data_by_chunk, alpha=0.5, bins=100)
        # plt.hist(impaired_data_by_chunk, alpha=0.5, bins=100)
        plt.hist(delta_data)
        plt.title(f"{condition} {test_column}\npvalue: {pvalue:.8f}\nmean diff: {mean_diff}")
        plt.xlabel("difference")
        plt.ylabel("frequency")
        plt.savefig(os.path.join(condition_out_dir, f"{pvalue:.8f}_{test_column}.png"))
        plt.close("all")
    return results


if __name__ == "__main__":
    main()
