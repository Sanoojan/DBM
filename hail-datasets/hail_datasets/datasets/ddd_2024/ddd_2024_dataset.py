import os
import pandas as pd
import numpy as np
import random
import copy
import csv

from ddd_2024_chunk import ChunkFactory
from ddd_2024_scenario import Scenario
from ddd_2024_participant import Participant
from ddd_2024_sample import Sample, CollatedSample
from ddd_2024_features.tabular_feature import TabularFeatureNormStats
from parsers.crash_counts import CrashStats
from parsers.dataset_anomalies import DatasetAnomalies


class FoldConfig():  
    def __init__(self, name="participant_name", num_splits=5, split_train_val=True):
        self.name = name
        self.num_splits = num_splits
        self.split_train_val = split_train_val
    
    def filter_table(self, df, test_split_index, fold_name, random_seed=42):
        # Reset random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get groups of values
        unique_values = df[self.name].unique()
        try:
            unique_values = unique_values[np.logical_not(np.isnan(unique_values))]
        except:
            pass
        random.shuffle(unique_values)
        splits = [unique_values[offset::self.num_splits] for offset in range(self.num_splits)]

        # Determine values for current fold
        if fold_name == "test":
            current_values = splits[test_split_index]
        else:
            del splits[test_split_index]
            if self.split_train_val:
                val_split_index = test_split_index if test_split_index < self.num_splits - 1 else 0
                if fold_name == "val":
                    current_values = splits[val_split_index]
                else:
                    del splits[val_split_index]
                    current_values = np.concatenate(splits)
            else:
                current_values = np.concatenate(splits)

        # Filter the scenario table for these values
        out = df[df[self.name].isin(current_values)]
        return out


class DDD2024DatasetConfig():
    def __init__(self):
        self.base_path = "dataset/Vehicle/No-Video/"  # The base path used by all other dataset paths - leave blank if no shared root
        self.index_relative_path = "Resampled_previous_10"  # The relative path to use for indexing the dataset and getting scenario and participant metadata
        self.folds_path = os.path.join(os.path.dirname(__file__), "folds.csv")  # The path to the csv file containing the folds when using the "fixed" fold_name setting
        
        self.experiment_version = None  # The experiment version to use (1 or 2; if None, use both)
        self.scenario_name_filter = ".*"  # Regex filter used to subselect scenario names
        self.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]
        self.use_participants = None
        self.downsample = None  # The percent of data downsampling to apply (if None, no downsampling is used)
        self.scenario_remove_anomalies = True  # If True, remove entire scenarios from the dataset if they contain anomalies
        self.exclude_cd = []  # The types of CD to exclude (nback_task, statement_task, no_task)
        self.exclude_intoxicated = None  # The type of intoxication to exclude (None - exclude nothing)
        self.train_downsample_intox = None  # The percent of intoxicated data downsampling to apply in the training dataset (if None, no downsampling is used)
        self.exclude_baseline = False
        
        self.chunk_strategy = "end"  # "full", "random", "start", or "end" - how to sample chunks from scenarios
        self.chunk_duration = 10.  # Duration of the chunk
        self.chunks_per_scenario = 1  # Maximum number of chunks to extract from each scenario - If None, will extract variable number based on scenario duration
        self.chunk_start_offset = 0.  # Seconds from the start of the scenario to not include
        self.chunk_end_offset = 0.  # Seconds from the end of the scenario to not include
        self.chunk_fps = None  # FPS to use when creating chunks - uses max feature FPS if not defined (requires at least one feature to have FPS)
        self.cache_chunks = False  # Whether or not to reuse selected chunks on future epochs
        self.chunk_remove_anomalies = 0.  # If chunk overlaps with anomaly longer than this percent amount, remove (if None, do not remove)
        # Paramters for chunk_strategy "start" or "end"
        self.chunk_stride = 5.  # Time offset between chunks in seconds
       
        self.random_seed = 42
        self.fold_configs = []
        self.test_split_index = 0
        self.fold_name = "all"
        
        self.features = {}  # The dictionary of features loaded by the dataset
        # Example
        # {
        #     "tobii": {
        #         "relative_path": "Resampled_previous_10",
        #         "file_name": "experiment_tobii_frame",
        #         "type": "tabular",
        #         "fps": 10,
        #         "columns": ["tobii_left_eye_gaze_pt_validity",]
        #     },
        #     "carla_actors": {
        #         "relative_path": "Resampled_previous_10",
        #         "file_name": "carla_actor_list",
        #         "type": "object_attributes",
        #         "object_id_column": "carla_actor_id",
        #     },
        #     "carla_tracks": {
        #         "relative_path": "Resampled_previous_10",
        #         "file_name": "carla_objects",
        #         "type": "object_tracks",
        #         "fps": 10,
        #         "object_id_column": "carla_objects_id",
        #     },
        #     "cam_front": {
        #         "relative_path": "Resampled_previous_10",
        #         "file_name": "cam_front",
        #         "type": "video",
        #         "fps": 10,
        #         "resolution": (960//2, 400//2, 3),
        #     }
        # }
        
        # Raw video resolutions
        # "cam_front": (960, 400, 3)
        # "cam_depth": (960, 400, 1)
        # "cam_seg": (960, 400, 1)
        # "cam_face": (960, 960, 3)


class DDD2024Dataset():
    def __init__(self, config, train_dataset=None):
        self.config = config
        self.config.base_path = os.path.expanduser(self.config.base_path)

        # Reset random seed
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Load the list of participants and scenarios
        self.participants = {}
        self.scenario_index = []
        base_index_dir = os.path.join(self.config.base_path, self.config.index_relative_path, "Participants")
        participant_names = os.listdir(base_index_dir)
        random.shuffle(participant_names)
        for participant_name in participant_names:
            # Check if participant is removed
            if participant_name in self.config.remove_participants:
                continue
            if self.config.use_participants is not None and participant_name not in self.config.use_participants:
                continue

            # Create the participant
            new_participant = Participant(participant_name, base_index_dir)
            self.participants[participant_name] = new_participant

            # Load the scenario index
            scenario_index_path = os.path.join(new_participant.index_path, "scenario_index.csv")
            participant_scenario_index = pd.read_csv(scenario_index_path, index_col=False)
            participant_scenario_index["participant_name"] = participant_name
            participant_scenario_index["experiment_version"] = 1 if (participant_name[0] == "P") else 2
            participant_scenario_index["intoxicated"] = (participant_name[0] == "P") and (participant_scenario_index["round"] == "R2")

            # Append the scenarios
            self.scenario_index.append(participant_scenario_index)
            
        # Create the scenario dataframe
        self.scenario_index = pd.concat(self.scenario_index)

        # Select only certain experiment version
        if self.config.experiment_version is not None:
            self.scenario_index = self.scenario_index[self.scenario_index['experiment_version'] == self.config.experiment_version]
        
        # Filter the scenarios actually used
        self.scenario_index = self.scenario_index[self.scenario_index['scenario_name'].str.match(self.config.scenario_name_filter)]

        # Parse other files for additional metadata
        self.scenario_index.reset_index(inplace=True)
        crash_stats = CrashStats(self.config.base_path)
        anomalies = DatasetAnomalies(self.config.base_path)
        new_metas = []
        for index, row in self.scenario_index.iterrows():
            new_meta = {}
            new_meta["scenario_index"] = index
            new_meta["number_all_crashes"] = crash_stats.get_number_all_crashes(row["participant_name"], row["scenario_name"])
            new_meta["number_hazard_crashes"] = crash_stats.get_number_hazard_crashes(row["participant_name"], row["scenario_name"])
            new_meta["anomalies"] = anomalies.get_anomalies(row["participant_name"], row["scenario_name"])
            new_metas.append(new_meta)
        new_metas = pd.DataFrame(new_metas).set_index("scenario_index")
        self.scenario_index = pd.concat([self.scenario_index, new_metas], axis="columns")

        # Remove scenarios with anomalies, if requested
        if self.config.scenario_remove_anomalies:
            self.scenario_index = self.scenario_index[self.scenario_index['anomalies'].apply(len) == 0]

        # Filter out certain types of CD
        if self.config.fold_name == "train" and len(self.config.exclude_cd) > 0:
            self.scenario_index = self.scenario_index[np.logical_not(self.scenario_index["cognitive_task"].isin(self.config.exclude_cd))]

        # Filter out certain types of intoxication
        if self.config.fold_name == "train" and self.config.exclude_intoxicated is not None:
            self.scenario_index = self.scenario_index[np.logical_not(self.scenario_index["intoxicated"]==self.config.exclude_intoxicated)]

        # Filter out baseline data
        if self.config.fold_name == "train" and self.config.exclude_baseline:
            is_impaired = np.logical_or(self.scenario_index["intoxicated"], self.scenario_index["cognitive_task"] != "no_task")
            self.scenario_index = self.scenario_index[is_impaired]

        # Add the fixed_fold to the scenario index
        self.scenario_index["fixed_fold"] = np.nan
        with open(self.config.folds_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for fold_on, fold_participants in enumerate(csvreader):
                self.scenario_index.loc[self.scenario_index["participant_name"].isin(fold_participants), "fixed_fold"] = fold_on

        # Filter by the fold setup
        if self.config.fold_name != "all":
            cur_split_index = self.config.test_split_index
            for fold_config in self.config.fold_configs:
                config_index = cur_split_index % fold_config.num_splits
                cur_split_index = cur_split_index // fold_config.num_splits
                self.scenario_index = fold_config.filter_table(self.scenario_index, config_index, self.config.fold_name, self.config.random_seed)
        
        # Downsample a smaller dataset
        if self.config.downsample is not None and self.config.downsample < 1.:
            self.scenario_index = self.scenario_index.sample(frac=self.config.downsample, replace=False)

        # Downsample the training intoxicated data
        if self.config.fold_name == "train" and self.config.train_downsample_intox is not None and self.config.train_downsample_intox < 1.:
            intox_df = self.scenario_index[self.scenario_index["intoxicated"]]
            other_df = self.scenario_index[np.logical_not(self.scenario_index["intoxicated"])]
            intox_df = intox_df.sample(frac=self.config.train_downsample_intox, replace=False)
            self.scenario_index = pd.concat([intox_df, other_df], axis="rows")

        # Convert to scenario objects
        self.scenario_index.reset_index(inplace=True)
        self.scenarios = Scenario.list_from_table(self.scenario_index)
        
        # Setup chunking
        self.chunk_factory = ChunkFactory(self.config, self.scenarios)

        # NORMALIZATION        
        # If a training dataset exists, load previous population stats
        if train_dataset is None:
            # Only load features that need normalization
            self.tabular_norm_stats = TabularFeatureNormStats()
            norm_feature_config = {key: value for (key, value) in self.config.features.items()
                                   if ("normalize" in value and value["normalize"] != "none") or
                                   ("normalize_aggregate" in value and value["normalize_aggregate"] != "none")}
        else:
            # Only load features that need subject normalization
            self.tabular_norm_stats = copy.deepcopy(train_dataset.tabular_norm_stats)
            norm_feature_config = {key: value for (key, value) in self.config.features.items()
                                   if ("normalize" in value and value["normalize"] == "subject") or 
                                   ("normalize_aggregate" in value and value["normalize_aggregate"] == "subject")}

        # Run through all dataset samples loading features needing normalization (if any)
        if len(norm_feature_config) > 0:
            # Temporarily only consider normalized feature sets
            cache_feature_config = copy.deepcopy(self.config.features)
            self.config.features = norm_feature_config

            # Iterate through all samples to calculate normalization parameters
            print("Iterating through all samples to calculate normalization parameters")
            for sample in self:
                pass

            # Restore full feature set
            self.tabular_norm_stats.finalize()
            self.config.features = cache_feature_config

        # Summary messsage on creation
        num_chunks = len(self.chunk_factory)
        num_scenarios = len(self.scenario_index.groupby(["scenario_name", "participant_name"]).size())
        num_participants = len(self.scenario_index["participant_name"].unique())
        print(f"Constructed dataset contains {num_participants} participants, {num_scenarios} scenarios, and {num_chunks} chunks.")

    def __len__(self):
        return len(self.chunk_factory)

    def __getitem__(self, item):
        # Sample scenario, participant, and chunk
        chunk_index = self.chunk_factory.get_index(item)
        scenario = self.scenarios[chunk_index.scenario_number]
        participant_name = scenario.participant_name
        participant = self.participants[participant_name]
        chunk = self.chunk_factory.get_chunk(chunk_index, scenario)
        sample = Sample(scenario, participant, chunk, self)            
        return sample

    def collate(self, list_of_samples):
        return CollatedSample(list_of_samples, self)
