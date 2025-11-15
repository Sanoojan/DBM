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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="~/data/IDD/Processed",
        help="path to the top-level of the input resampled data directory",
    )
    args = parser.parse_args()

    # Specify input features needed
    config = DBM_Dataset_Config()
    config.base_path = args.in_dir
    config.index_relative_path = "Resampled_previous_10"

    config.features = {
        "ego_tracks": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "tabular",
            "fps": 10,
            "columns": [
                "longitudinal_speed",
            ],
        },
    }

    config.scenario_name_filter = "driving/[0-9].*"
    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
    config.scenario_remove_anomalies = False

    # Use 60 second chunks, offset 30 seconds from the end to exclude hazard
    config.chunk_strategy = "full"

    # Load all data
    dataset = DBM_Dataset(config)

    # Loop through and calculate statistics
    tallys = {}
    totals = {}
    totals["Count"] = 0
    totals["Distance"] = 0
    totals["Hazards"] = 0
    for sample_on, sample in enumerate(dataset):
        # Scenario metadata
        intoxicated = sample.scenario.intoxicated
        cd = sample.scenario.cognitive_task != "no_task"
        condition = f"intox {intoxicated} cd {cd}"
        if condition not in tallys:
            tallys[condition] = {}
            tallys[condition]["Count"] = 0
            tallys[condition]["Distance"] = 0
            tallys[condition]["Hazards"] = 0
        tallys[condition]["Count"] += 1
        totals["Count"] += 1

        # Crashes
        crashes = sample.scenario.number_hazard_crashes
        if not np.isnan(crashes):
            tallys[condition]["Hazards"] += 1
            totals["Hazards"] += 1

        # Distance
        fps = dataset.config.features["ego_tracks"]["fps"]
        distance = np.sum(sample.features["ego_tracks"].table.to_numpy()) / (fps * 1000)
        tallys[condition]["Distance"] += distance
        totals["Distance"] += distance

    for condition, data in tallys.items():
        print(condition, data)
    print(totals)


if __name__ == "__main__":
    main()
