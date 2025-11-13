import os
import numpy as np

from ddd_2024_features.tabular_feature import TabularFeature, CollatedTabularFeature
from ddd_2024_features.object_attributes_feature import ObjectAttributesFeature, CollatedObjectAttributesFeature
from ddd_2024_features.object_tracks_feature import ObjectTracksFeature, CollatedObjectTracksFeature
from ddd_2024_features.video_feature import VideoFeature, CollatedVideoFeature
from ddd_2024_features.dictionary_array_feature import DictionaryArrayFeature, CollatedDictionaryArrayFeature
from ddd_2024_chunk import CollatedChunk
from ddd_2024_scenario import CollatedScenario
from ddd_2024_participant import CollatedParticipant

class Sample():
    def __init__(self, scenario, participant, chunk, dataset):
        # Save metadata
        self.dataset = dataset
        self.scenario = scenario
        self.participant = participant
        participant_name = self.participant.name
        self.chunk = chunk
        self.relative_path = os.path.join("Participants", participant_name, scenario.scenario_path)
 
        # Loop through features and load data
        self.features = {}
        self.feature_resample_times = {}
        self.feature_num_frames = {}
        for feature_name, feature_attributes in dataset.config.features.items():
            if "file_name" not in feature_attributes:
                feature_attributes["file_name"] = feature_name
            
            base_participant_fps_dir = os.path.join(dataset.config.base_path, feature_attributes["relative_path"], "Participants")            
            participant_fps_path = os.path.join(base_participant_fps_dir, participant_name)
            full_scenario_path = os.path.join(participant_fps_path, scenario.scenario_path)
            
            # Get the resample times if has a frame rate
            if "fps" in feature_attributes:
                fps = feature_attributes["fps"]
                if fps not in self.feature_resample_times:
                    resample_times = scenario.get_resample_times(fps)
                    resample_times = chunk.filter_resample_times(resample_times)                    
                    self.feature_resample_times[fps] = resample_times
                    self.feature_num_frames[fps] = len(resample_times)

            # Load based on the type of feature
            try:
                if feature_attributes["type"] == "tabular":
                    self.features[feature_name] = TabularFeature(feature_name, feature_attributes, full_scenario_path, self)
                elif feature_attributes["type"] == "object_attributes":
                    self.features[feature_name] = ObjectAttributesFeature(feature_name, feature_attributes, full_scenario_path, self)
                elif feature_attributes["type"] == "object_tracks":
                    self.features[feature_name] = ObjectTracksFeature(feature_name, feature_attributes, full_scenario_path, self)
                elif feature_attributes["type"] == "video":
                    self.features[feature_name] = VideoFeature(feature_name, feature_attributes, full_scenario_path, self)
                elif feature_attributes["type"] == "dictionary_array":
                    self.features[feature_name] = DictionaryArrayFeature(feature_name, feature_attributes, full_scenario_path, self)
                else:
                    print("WARNING: Feature type {:s} doesn't exist. Not extracting {:s} feature.".format(feature_attributes["type"], feature_name))
            except FileNotFoundError:
                pass

    def close(self):
        for feature_name, feature in self.features.items():
            feature.close()

    def __str__(self):
        feature_list = ", ".join(self.features.keys())
        return f"{self.scenario}, {self.chunk}, Features: ({feature_list})"
    
class CollatedSample():
    def __init__(self, list_of_samples, dataset):
        # Setup the batch and sample metadata
        self.config = dataset.config
        self.batch_size = len(list_of_samples)
        self.scenario = CollatedScenario([x.scenario for x in list_of_samples], dataset.config)
        self.participant = CollatedParticipant([x.participant for x in list_of_samples], dataset.config)
        self.relative_path = np.array([x.relative_path for x in list_of_samples])

        # Collate chunk data
        self.chunk = CollatedChunk([x.chunk for x in list_of_samples], dataset.config)

        # Collate frame rate metadata
        self.feature_resample_times = {}
        self.feature_num_frames = {}
        for fps in list_of_samples[0].feature_resample_times.keys():
            self.feature_resample_times[fps] = np.array([x.feature_resample_times[fps] for x in list_of_samples])
            self.feature_num_frames[fps] = list_of_samples[0].feature_num_frames[fps]  # Should be same across batch

        # Collate features
        self.features = {}
        for feature_name, feature_attributes in dataset.config.features.items():
            # Collate based on the type of data
            feature_attributes["name"] = feature_name
            list_of_features = [x.features[feature_name] if feature_name in x.features else None for x in list_of_samples]
            if feature_attributes["type"] == "tabular":
                self.features[feature_name] = CollatedTabularFeature(feature_name, feature_attributes, list_of_features, self)
            elif feature_attributes["type"] == "object_attributes":
                self.features[feature_name] = CollatedObjectAttributesFeature(feature_name, feature_attributes, list_of_features, self)
            elif feature_attributes["type"] == "object_tracks":
                self.features[feature_name] = CollatedObjectTracksFeature(feature_name, feature_attributes, list_of_features, self)
            elif feature_attributes["type"] == "video":
                self.features[feature_name] = CollatedVideoFeature(feature_name, feature_attributes, list_of_features, self)
            elif feature_attributes["type"] == "dictionary_array":
                self.features[feature_name] = CollatedDictionaryArrayFeature(feature_name, feature_attributes, list_of_features, self)
            else:
                print(
                    "WARNING: Feature type {:s} doesn't exist. Not extracting {:s} feature.".format(
                        feature_attributes["type"], feature_name
                    )
                )
    