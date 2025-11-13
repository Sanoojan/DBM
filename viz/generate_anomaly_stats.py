#!/usr/bin/env python3
import numpy as np

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config


def main():
    # Load the hazard dataset
    config = DBM_Dataset_Config()

    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
    config.scenario_name_filter = "driving/([0-9]).*"
    # config.scenario_name_filter = "driving/(3c|5).*"
    config.features = {}
    config.scenario_remove_anomalies = False
    config.chunk_remove_anomalies = None
    dataset = DBM_Dataset(config)

    # Determine the totals
    total_duration = 0.0
    total_floating_cars = 0.0
    for sample in dataset:
        anomaly_ranges = sample.scenario.anomalies
        for anomaly_range in anomaly_ranges:
            total_floating_cars += (anomaly_range["end_ns"] - anomaly_range["start_ns"]) / 1e9
        total_duration += sample.scenario.duration_s
    print(total_floating_cars, total_duration, total_floating_cars / total_duration)


if __name__ == "__main__":
    main()
