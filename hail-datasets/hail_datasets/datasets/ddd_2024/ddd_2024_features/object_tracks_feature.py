import numpy as np

from ddd_2024_features.base_feature import BaseFeature, BaseCollatedFeature
from ddd_2024_features.common import get_matrix_from_tabular, format_columns


# Object tracks - data variable in length over time and variable in number of objects
class ObjectTracksFeature(BaseFeature):
    def __init__(self, feature_name, feature_attributes, full_scenario_path, sample):
        super().__init__(feature_name, feature_attributes)
        
        # Load track data
        csv_path = self.get_feature_path(full_scenario_path, feature_attributes["file_name"] + ".csv")
        rename_dict = {feature_attributes["object_id_column"]: "object_id"} if "object_id_column" in feature_attributes else None
        csv_columns = feature_attributes["columns"] if "columns" in feature_attributes else None
        track_data = self.load_dataframe(csv_path, index=["resampled_epoch_ns", "object_id"], column_list=csv_columns, rename_dict=rename_dict)
                        
        # Filter by chunk times - copy if not in cache
        track_data = sample.chunk.filter_dataframe(track_data)
        
        # Get list of remaining objects
        object_ids = track_data.index.unique(level="object_id")
        
        # Load all actors into dictionary
        all_tracks = {}
        all_relevance = []
        for object_id in object_ids:
            # Load the tracks
            object_tracks = track_data.loc[(slice(None), object_id), :].copy()
            if object_tracks.shape[0] == 0:
                continue
            object_tracks.reset_index(inplace=True)
            object_tracks.drop(columns="object_id", inplace=True)
            object_tracks.set_index("resampled_epoch_ns", inplace=True, drop=True)            

            # If relevance provided, determine minimum of the relevance over entire chunk
            if "relevance" in object_tracks.columns:
                # Calculate relevance
                object_relevance = np.nanmin(object_tracks["relevance"].to_numpy())
                all_relevance.append({"relevance": object_relevance, "object_id": int(object_id)})

            # Save object tracks to dictionary
            all_tracks[int(object_id)] = object_tracks

        # Sort tracks by dictionary
        if len(all_relevance) > 0:
            all_relevance.sort(key=lambda d: d["relevance"])
            self.tracks = {}
            for relevance_pair in all_relevance:
                self.tracks[relevance_pair["object_id"]] = all_tracks[relevance_pair["object_id"]]
        else:
            self.tracks = all_tracks


class CollatedObjectTracksFeature(BaseCollatedFeature):
    # Populates the following:
    # self.object_id (batch_size, max_objects)
    # self.tracks (batch_size, max_objects, num_frames, num_columns)
    # self.tracks_valid (batch_size, max_objects, num_frames)
    # self.tracks_columns
    
    def __init__(self, feature_name, feature_attributes, list_of_features, collated_sample):
        super().__init__(feature_name, feature_attributes)

        batch_size = collated_sample.batch_size
        fps = collated_sample.config.features[feature_name]["fps"]
        num_frames = collated_sample.feature_num_frames[fps]
        track_columns = format_columns(feature_attributes)
        if "max_objects" not in feature_attributes:
            raise Exception(f"Must declare max objects for feature {feature_name} when using collation.")
        max_objects = feature_attributes["max_objects"]

        # Initialize outputs
        self.object_id = np.zeros((batch_size, max_objects), dtype=np.int64)
        self.tracks_valid = np.zeros((batch_size, max_objects, num_frames))
        self.tracks = np.zeros((batch_size, max_objects, num_frames, len(track_columns)))
        self.tracks_columns = track_columns
        self.file_valid = np.zeros((batch_size,))

        # Loop through samples and add the tracks
        for sample_on, feature_data in enumerate(list_of_features):
            if feature_data is not None:
                self.file_valid[sample_on] = 1
                track_dict = feature_data.tracks
                for object_index, (object_id, object_tracks) in enumerate(track_dict.items()):
                    if object_index >= max_objects:
                        break
                    self.object_id[sample_on, object_index] = object_id
                    track_matrix_data = get_matrix_from_tabular(
                        object_tracks, track_columns, collated_sample.feature_resample_times[fps][sample_on]
                    )
                    self.tracks_valid[sample_on, object_index] = np.logical_not(
                        np.any(np.isnan(track_matrix_data), axis=-1)
                    )
                    track_matrix_data[np.isnan(track_matrix_data)] = 0.0
                    self.tracks[sample_on, object_index] = track_matrix_data
