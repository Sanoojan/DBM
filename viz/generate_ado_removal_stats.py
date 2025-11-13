#!/usr/bin/env python3
import numpy as np

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config


def main():
    # Load the hazard dataset
    config = DBM_Dataset_Config()

    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
    config.scenario_name_filter = "driving/([0-9]).*"
    # config.scenario_name_filter = "driving/(3c|5).*"
    config.features = {
        "ado_removal_stats": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "object_attributes",
        },
    }
    config.chunk_strategy = "full"
    dataset = DBM_Dataset(config)

    # Determine the totals
    totals = {}
    totals["scenarios"] = 0
    totals["removals"] = 0
    totals["duration"] = 0
    totals["visible"] = 0
    for sample in dataset:
        if "ado_removal_stats" not in sample.features:
            continue
        removal_stats = sample.features["ado_removal_stats"].attributes
        removal_causes = removal_stats["cause"].value_counts()
        for cause, count in removal_causes.items():
            if cause not in totals:
                totals[cause] = 0
            totals[cause] += count
            totals["removals"] += count
        totals["scenarios"] += 1
        totals["duration"] += sample.scenario.duration_s
        totals["visible"] += int(
            np.sum(removal_stats["on_screen"] & removal_stats["cause"].isin(["still", "hazard", "unknown"]))
        )
    print(totals)


if __name__ == "__main__":
    main()
