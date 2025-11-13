#!/usr/bin/env python3
import numpy as np

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


def main():
    # Load the entire dataset
    config = DBM_Dataset_Config()
    config.scenario_name_filter = ".*"
    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
    dataset = DBM_Dataset(config)

    # Setup scenario name to filter dict
    scenario_filters = {
        "Calibration": "tobii_calibration",
        "Stationary": "stationary_tasks.*",
        "Practice": "driving/practice",
        "Hazards": "driving/[0-9].*",
        "All": ".*",
        "Driving": "driving/.*",
    }
    plus_minus = "\u00B1"

    # Setup logger
    logger = Logger()

    # Header for driving data
    scenario_types = ["Practice", "Hazards", "Driving"]
    logger.print("Driving Data")
    header_row = "Intoxicated,Cog. Distracted"
    for scenario_type in scenario_types:
        header_row += f",{scenario_type}"
    logger.print(header_row)

    # Each row filters over cognitive task and intoxication
    for intoxicated_filter in [False, True, None]:
        for cd_filter in [False, True, None]:
            # Form impairment subset
            impairment_subset = dataset.scenario_index
            if intoxicated_filter is not None:
                impairment_subset = impairment_subset[impairment_subset["intoxicated"] == intoxicated_filter]
            if cd_filter is not None:
                impairment_subset = impairment_subset[(impairment_subset["cognitive_task"] != "no_task") == cd_filter]

            # First columns describe the impairment_subset conditions
            row_string = f"{impaired_condition_to_text(intoxicated_filter)},{impaired_condition_to_text(cd_filter)}"

            # Each grouped column is based on filtering the scenario type
            for scenario_type in scenario_types:
                scenario_filter = scenario_filters[scenario_type]
                scenario_subset = impairment_subset[impairment_subset["scenario_name"].str.match(scenario_filter)]

                # Print three stats for each scenario group
                num_scenarios = scenario_subset.shape[0]
                num_hours = (scenario_subset["duration_s"].sum() / 60.0) / 60.0
                row_string += f",{num_scenarios} ({num_hours:.1f})"

            # Print row
            logger.print(row_string)

    # Separator
    logger.print()

    # Header for task data
    scenario_types = ["Calibration", "Stationary"]
    logger.print("Task Data")
    header_row = "Intoxicated"
    for scenario_type in scenario_types:
        header_row += f",{scenario_type}"
    logger.print(header_row)

    # Each row filters over intoxication
    for intoxicated_filter in [False, True, None]:
        # Form impairment subset
        impairment_subset = dataset.scenario_index
        if intoxicated_filter is not None:
            impairment_subset = impairment_subset[impairment_subset["intoxicated"] == intoxicated_filter]

        # First columns describe the impairment_subset conditions
        row_string = f"{impaired_condition_to_text(intoxicated_filter)}"

        # Each grouped column is based on filtering the scenario type
        for scenario_type in scenario_types:
            scenario_filter = scenario_filters[scenario_type]
            scenario_subset = impairment_subset[impairment_subset["scenario_name"].str.match(scenario_filter)]

            # Print three stats for each scenario group
            num_scenarios = scenario_subset.shape[0]
            num_hours = (scenario_subset["duration_s"].sum() / 60.0) / 60.0
            row_string += f",{num_scenarios} ({num_hours:.1f})"

        # Print row
        logger.print(row_string)

    # Separator
    logger.print()

    # Header for participant data
    logger.print("Participant Completion Data")
    header_row = "Intoxicated,Started,Four Hazards,All Hazards"
    logger.print(header_row)

    # Each row filters over intoxication
    for intoxicated_filter in [False, True, None]:
        # Form impairment subset
        impairment_subset = dataset.scenario_index
        if intoxicated_filter is not None:
            impairment_subset = impairment_subset[
                ((impairment_subset["participant_name"].str[0] == "P") == intoxicated_filter)
            ]

        # First columns describe the impairment_subset conditions
        row_string = f"{impaired_condition_to_text(intoxicated_filter)}"
        row_string += "," + str(len(impairment_subset["participant_name"].unique()))

        # Only consider hazards
        hazard_subset = impairment_subset[impairment_subset["scenario_name"].str.match(scenario_filters["Hazards"])]
        hazard_counts = hazard_subset.groupby("participant_name").agg("count")
        row_string += "," + str((hazard_counts["scenario_index"] >= 4).sum())
        row_string += "," + str((hazard_counts["scenario_index"] >= 8).sum())

        # Print row
        logger.print(row_string)

    # Separator
    logger.print()

    # Header for total duration data
    scenario_types = ["Calibration", "Stationary", "Practice", "Hazards"]
    logger.print("Total Durations")
    header_row = "Scenario,Number Iterations,Mean Duration (Minutes),Total Duration (Minutes)"
    logger.print(header_row)

    # Each row filters over scenario type
    for scenario_type in scenario_types:
        row_string = scenario_type
        scenario_filter = scenario_filters[scenario_type]
        scenario_subset = dataset.scenario_index[dataset.scenario_index["scenario_name"].str.match(scenario_filter)]

        # Filter to subjects who completed all scenarios
        completed_counts = scenario_subset.groupby("participant_name").agg("count")
        most_completed = completed_counts["scenario_index"].max()
        completed_subjects = completed_counts[completed_counts["scenario_index"] == most_completed].index.values
        scenario_subset = scenario_subset[scenario_subset["participant_name"].isin(completed_subjects)]

        # Get aggregate stats over subjects completing the particular scenario
        mean_duration = scenario_subset["duration_s"].mean() / 60.0
        std_duration = scenario_subset["duration_s"].std() / 60.0
        row_string += f" {mean_duration:.1f} {plus_minus} {std_duration:.1f}"

        # Print row
        logger.print(row_string)

    # Close logger
    logger.close()


if __name__ == "__main__":
    main()
