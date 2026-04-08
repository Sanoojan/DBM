import csv
import os

import numpy as np


class DatasetAnomalies:
    def __init__(self, base_dataset_path):
        self.all_anomaly_map = {}
        anomaly_stats_path = os.path.join(base_dataset_path, "idd_annotation.csv")
        with open(anomaly_stats_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            next(csv_reader)
            for row in csv_reader:
                participant = row[0]
                if participant[0] not in ["P", "7"]:
                    continue
                scenario_names = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]

                all_anomalies = row[1 : 1 + 8]
                self.all_anomaly_map[participant] = {}
                for scenario_name, annotation in zip(scenario_names, all_anomalies):
                    out_list = []
                    for float_car_range in annotation.split(","):
                        out_range = tuple([int(x) for x in float_car_range.strip().split("-") if x != ""])
                        if len(out_range) == 2:
                            out_list.append({"start_ns": int(out_range[0]*1e9), "end_ns": int(out_range[1]*1e9)})
                    self.all_anomaly_map[participant][scenario_name] = out_list

    def get_anomalies(self, participant, scenario_name):
        if "driving" not in scenario_name:
            return []
        scenario_name = scenario_name.split("/")[1].split("-")[0]
        if participant not in self.all_anomaly_map:
            return []
        if scenario_name not in self.all_anomaly_map[participant]:
            return []
        return self.all_anomaly_map[participant][scenario_name]


def main():
    crash_stats = DatasetAnomalies("dataset/Vehicle/No-Video")
    print(crash_stats.get_anomalies("7207", "2"))


if __name__ == "__main__":
    main()
