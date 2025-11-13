import csv
import os

import numpy as np


class CrashStats:
    def __init__(self, base_dataset_path):
        self.all_crash_map = {}
        self.hazard_crash_map = {}
        crash_stats_path = os.path.join(base_dataset_path, "idd_annotation.csv")
        with open(crash_stats_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            next(csv_reader)
            for row in csv_reader:
                participant = row[0]
                if participant[0] not in ["P", "7"]:
                    continue
                scenario_names = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]

                all_crashes = row[17 : 17 + 8]
                self.all_crash_map[participant] = {}
                for scenario_name, annotation in zip(scenario_names, all_crashes):
                    if annotation.strip() == "":
                        self.all_crash_map[participant][scenario_name] = 0
                    else:
                        self.all_crash_map[participant][scenario_name] = annotation.count(",") + 1

                hazard_crashes = row[9 : 9 + 8]
                self.hazard_crash_map[participant] = {}
                for scenario_name, annotation in zip(scenario_names, hazard_crashes):
                    if "fail" in annotation:
                        self.hazard_crash_map[participant][scenario_name] = -1
                    elif "crash" in annotation:
                        self.hazard_crash_map[participant][scenario_name] = 1
                    else:
                        self.hazard_crash_map[participant][scenario_name] = 0

    def get_number_all_crashes(self, participant, scenario_name):
        if "driving" not in scenario_name:
            return 0
        scenario_name = scenario_name.split("/")[1].split("-")[0]
        if participant not in self.all_crash_map:
            return -1
        if scenario_name not in self.all_crash_map[participant]:
            return -1
        return self.all_crash_map[participant][scenario_name]

    def get_number_hazard_crashes(self, participant, scenario_name):
        if "driving" not in scenario_name:
            return 0
        scenario_name = scenario_name.split("/")[1].split("-")[0]
        if participant not in self.hazard_crash_map:
            return -1
        if scenario_name not in self.hazard_crash_map[participant]:
            return -1
        return self.hazard_crash_map[participant][scenario_name]


def main():
    crash_stats = CrashStats("~/data/IDD/Processed")
    print(crash_stats.get_number_crashes("7207", "3c"))


if __name__ == "__main__":
    main()
