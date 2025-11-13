"""
Dataset utilities
"""

import glob
import os

import pandas as pd


def get_scenario_list(exp="24-d-07"):
    if exp == "24-d-07":
        return [
            "1a-pedestrian_pop_out",
            "2-vehicle_pop_out",
            "2b-vehicle_door_open_hazard",
            "3c-pedestrian_pop_out",
            "5-vehicle_run_stop",
            "6e-vehicle_run_red_light",
            "7a-vehicle_run_red_light",
            "8a-pedestrian_pop_out",
        ]
    else:
        raise NotImplementedError


def get_scenario_abbreviation_dict(version):
    """return a dictionary of abbreviated scenario names for cleaner figures"""
    if version == "v1":
        scenarios_abbreviated = {
            "R1-tobii_calibration": "R1-TC",
            "R1-stationary_tasks/choice_reaction": "R1-ST-CR",
            "R1-stationary_tasks/gaze_tracking": "R1-ST-GT",
            "R1-stationary_tasks/fixed_gaze": "R1-ST-FG",
            "R1-stationary_tasks/silent_reading": "R1-ST-SR",
            "R1-driving/practice/no_task": "R1-DP-C",  # C for Control
            "R1-driving/practice/nback_task": "R1-DP-NB",
            "R1-driving/practice/statement_task": "R1-DP-ST",
            "R1-driving/*-*/no_task": "R1-D-C",
            "R1-driving/*-*/nback_task": "R1-D-NB",
            "R1-driving/*-*/no_task-2": "R1-D-C-2",  # second occurrence gets -2
            "R1-driving/*-*/statement_task": "R1-D-ST",
            "R2-tobii_calibration": "R2-TC",
            "R2-stationary_tasks_recap/choice_reaction": "R2-ST-CR",
            "R2-stationary_tasks_recap/gaze_tracking": "R2-ST-GT",
            "R2-stationary_tasks_recap/fixed_gaze": "R2-ST-FG",
            "R2-stationary_tasks_recap/silent_reading": "R2-ST-SR",
            "R2-driving/*-*/no_task": "R2-D-C",
            "R2-driving/*-*/nback_task": "R2-D-NB",
            "R2-driving/*-*/no_task-2": "R2-D-C-2",  # second occurrence gets -2
            "R2-driving/*-*/statement_task": "R2-D-ST",
        }
        return scenarios_abbreviated
    else:
        raise ValueError


def fetch_participants(data_path, version="all"):
    """return a list of participants in the Participants/ path"""
    if version == "all":
        search_string = "*"
    elif version == "v1":
        search_string = "P7*"  # alcohol study
    elif version == "v2":
        search_string = "72*"
    participant_list = glob.glob(os.path.join(data_path, search_string))
    participant_list = sorted([p.split("/")[-1] for p in participant_list])
    return participant_list


def read_scenario_tables(data_path, participant_list):
    """read in a dict of key secnario tables from a given list of participants"""
    scenario_tables = {}
    for p in sorted(participant_list):
        csv_path = os.path.join(data_path, p, "key_scenario_timings.csv")
        scenario_tables[p] = pd.read_csv(csv_path)
    return scenario_tables


def get_key_actor_ids(carla_actor_list, hazard_name=None):
    """Given a carla actor list (and optional hazard name), return dict with id of ego (and hazard)"""

    actors = {}
    ego_vehicle_type = "vehicle.tri.motionsimvehicle_rwd"
    ego_info = {
        "1a-pedestrian_pop_out": (663.6864624023438, -6.40778923034668, 135.1903076171875),
        "2-vehicle_pop_out": (957.3306274414062, 250.47084045410156, 131.37347412109375),
        "2b-vehicle_door_open_hazard": (788.3154296875, 104.47952270507812, 134.13661193847656),
        "3c-pedestrian_pop_out": (533.6, 884.7, 120.9),
        "5-vehicle_run_stop": (-713.863525390625, 206.71876525878906, 169.02793884277344),
        "6e-vehicle_run_red_light": (13.44, -559.1, 160.45),
        "7a-vehicle_run_red_light": (13.44, -559.1, 160.45),
        "8a-pedestrian_pop_out": (146.0113983154297, -446.88861083984375, 153.09161376953125),
    }
    hazard_info = {
        "1a-pedestrian_pop_out": ("walker.pedestrian.0016", (9.61, 212.56, 154.85)),
        "2-vehicle_pop_out": ("vehicle.mercedes.coupe_2020", (-193.88, 271.70, 159.4)),
        "2b-vehicle_door_open_hazard": ("vehicle.mercedes.coupe_2020", (-33.17, 186.52, 153.65)),
        "3c-pedestrian_pop_out": ("walker.pedestrian.0016", (-621.68, 243.01, 169.98)),
        "5-vehicle_run_stop": (
            "vehicle.mercedes.coupe_2020",
            (51.23496627807617, -506.0608825683594, 155.83294677734375),
        ),
        "6e-vehicle_run_red_light": (
            "vehicle.mercedes.coupe_2020",
            (958.4266967773438, 389.3297424316406, 123.7259750366211),
        ),
        "7a-vehicle_run_red_light": ("vehicle.mercedes.coupe_2020", (765.47, 763.54, 118.48)),
        "8a-pedestrian_pop_out": ("walker.pedestrian.0013", (500.27, 852.25, 123.16)),
    }

    # find ego - there should only ever be one
    actors["ego"] = None
    mask = carla_actor_list["carla_actor_rolename"] == "hero"
    if sum(mask) == 0:
        print(f"No ego vehicle found (with carla_actor_rolename == 'hero').")
    elif sum(mask) > 1:
        print(f"{sum(mask)} ego vehicles found.")
    else:
        ego = carla_actor_list[mask].copy().reset_index(drop=True)
        ego = ego.loc[0].to_dict()
        actors["ego"] = ego["carla_actor_id"]

    # find hazard
    hazard_type, hazard_start_coords = hazard_info[hazard_name]
    mask = carla_actor_list["carla_actor_type"].str.contains(hazard_type, na=False)
    actors["hazard"] = None
    # find match which appears at same time (initialized from beginning)
    matches = carla_actor_list[mask].copy().reset_index(drop=True)
    if len(matches) > 0:
        for ii in range(len(matches)):
            m = matches.loc[ii].to_dict()
            if m["carla_actor_first_appearance log time"] - ego["carla_actor_first_appearance log time"] < 0.01:
                # TODO add pose/distance check if needed
                actors["hazard"] = m["carla_actor_id"]
                break
    if actors["hazard"] is None:
        print("Hazard not found")
    return actors


def read_exp_csv(participant_id, scenario, csv_name, exp_round=None, task=None, data_path=None):
    """
    Read a particular csv for a given participant and scenario

    Examples:
        df, csv_path = read_exp_csv("P703", "3c", "sim_bag/carla_objects.csv")
        df, csv_path = read_exp_csv("P703", "tobii_calibration", "sim_bag/*tobii_frame.csv", exp_round=1)
        df, csv_path = read_exp_csv("P703", "3c", "cam_front/frame_timing.csv")
        df, csv_path = read_exp_csv("P703", "3c", "sim_bag/carla_actor_list.csv", task="statement_task")
    """
    if not data_path:
        data_path = "/data/motion-simulator-logs/Processed/Clean/Participants/"

    # determine experiment round
    if exp_round:
        exp_round = "R" + str(exp_round)
    else:
        if scenario in ["choice_reaction", "fixed_gaze", "gaze_tracking", "silent_reading"]:
            raise TypeError("Must specify exp_round=1 or 2")
        else:
            exp_round = "*"

    # determine driving, stationary_*, or tobii_calibration
    if scenario in ["choice_reaction", "fixed_gaze", "gaze_tracking", "silent_reading"]:
        scenario_type = f"stationary*/{scenario}"
    elif scenario == "tobii_calibration":
        scenario_type = "tobii_calibration"
    else:
        if not task:
            task = "*"
        scenario_type = f"driving/{str(scenario)}*/{task}"

    csv_search_pattern = os.path.join(data_path, participant_id, exp_round, scenario_type, csv_name)
    csv_path = glob.glob(csv_search_pattern)
    if len(csv_path) == 1:
        csv_path = csv_path[0]
        df = pd.read_csv(csv_path)
    elif len(csv_path) == 0:
        print(f"Zero matches found for {csv_search_pattern}.")
        return None, csv_path
    else:
        print(f"Multiple csvs found: {csv_path}")
        return None, csv_path
    return df, csv_path
