import copy
import numpy as np
import pickle

from ddd_2024_features.base_feature import BaseFeature, BaseCollatedFeature 

   
# Object attributes - data constant over time, but variable in number of objects
class DictionaryArrayFeature(BaseFeature):
    def __init__(self, feature_name, feature_attributes, full_scenario_path, sample):
        super().__init__(feature_name, feature_attributes)
        
        # Load pickle data
        pkl_path = self.get_feature_path(full_scenario_path, feature_attributes["file_name"] + ".pkl")
        self.data = None
        with open(pkl_path, 'rb') as file:
            self.data = pickle.load(file)
        if self.data is None:
            raise FileNotFoundError()       
        
        # Verify input is dictionary
        if not isinstance(self.data, dict):
            raise Exception("Input data should be a dictionary.")
        
        # Verify input has time key
        if "resampled_epoch_ns" not in self.data:
            raise Exception("Data must contain resampled_epoch_ns key.")
        num_samples = len(self.data["resampled_epoch_ns"])

        # Verify that all inputs are numpy arrays and have first dimension of either 1 or num_samples
        for key, value in self.data.items():
            if not isinstance(value, np.ndarray):
                raise Exception(f"Entry {key} is not a numpy array.")
            if value.shape[0] != 1 and value.shape[0] != num_samples:
                raise Exception(f"Entry {key} has shape {value.shape} when first dimension should be 1 or {num_samples}.")
       
        # Filter by chunk times
        in_chunk = sample.chunk.times_in_chunk(self.data["resampled_epoch_ns"])
        for key in self.data.keys():
            if self.data[key].shape[0] != 1:
                self.data[key] = self.data[key][in_chunk]


class CollatedDictionaryArrayFeature(BaseCollatedFeature):
    def __init__(self, feature_name, feature_attributes, list_of_features, collated_sample):
        batch_size = collated_sample.batch_size
        fps = collated_sample.config.features[feature_name]["fps"]
        num_frames = collated_sample.feature_num_frames[fps]

        # Verify values and shapes are provided
        if "values" not in feature_attributes:
            raise Exception("Must provide values attribute when using collation on dictionary array feature.")

        # Set up each value
        self.data = {}
        for key, value_data in feature_attributes["values"].items():
            # Ensure data formatted as dictionary
            if isinstance(value_data, tuple) or isinstance(value_data, list):
                value_data = {"shape": value_data}
            if not isinstance(value_data, dict):
                raise Exception("Values must be either dictionary or shape.")

            # Initialize data output
            if "shape" not in value_data or (not isinstance(value_data["shape"], tuple) and not isinstance(value_data["shape"], list)):
                raise Exception("Shape must be specified as tuple or list.")
            value_shape = value_data["shape"]
            value_type = value_data["type"] if "type" in value_data else float
            default_for_type = np.nan if isinstance(value_type, np.floating) else 0
            value_default = value_data["default"] if "default" in value_data else default_for_type
            value_is_temporal = value_data["temporal"] if "temporal" in value_data else True
            final_value_shape = (batch_size, num_frames) + value_shape if value_is_temporal else (batch_size,) + value_shape            
            self.data[key] = np.full(final_value_shape, value_default, dtype=value_type)

            # Load all samples
            for i, sample_data in enumerate(list_of_features):
                if sample_data is not None:
                    self.data[key][i] = sample_data.data[key]
