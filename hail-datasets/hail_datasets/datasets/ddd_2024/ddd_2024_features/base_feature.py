import os
from abc import ABC
import copy
import pandas as pd


# Abstract class inherited by all features
class BaseFeature(ABC):
    df_cache = {}
    cache_features = True

    def __init__(self, name, feature_attributes):
        self.name = name
        self.fps = feature_attributes["fps"] if "fps" in feature_attributes else None

    def get_feature_path(self, full_scenario_path, rel_file_path):
        file_path = os.path.join(full_scenario_path, rel_file_path)
        if not os.path.exists(file_path):
            file_path = os.path.join(full_scenario_path, f"sim_bag/{rel_file_path}")
        return file_path

    @classmethod
    def load_dataframe(cls, csv_path, index=None, column_list=None, rename_dict=None):
        # Check cache
        if cls.cache_features and csv_path in cls.df_cache:
            return cls.df_cache[csv_path]

        # Read the file
        df = pd.read_csv(csv_path, engine="pyarrow")

        # Replace spaces in index and column names
        df.columns = df.columns.str.replace(' ', '_')
        if df.index.name is not None:
            df.index.name = df.index.name.replace(' ', '_')
        
        # Rename columns
        if rename_dict is not None:
            df.rename(columns=rename_dict, inplace=True)

        # Set up the index
        if index is not None:
            df.set_index(index, drop=True, inplace=True)

        # Filter columns used
        if column_list is not None:
            if not isinstance(column_list, list):
                column_list = [column_list,]
            df = df[column_list].copy()

        # Save to cache
        if cls.cache_features:
            cls.df_cache[csv_path] = df

        return df

    # Can be overriden by features to specify actions to take when a feature is unloaded
    def close(self):
        pass


class BaseCollatedFeature(ABC):
    def __init__(self, name, feature_attributes):
        self.name = name
        self.fps = feature_attributes["fps"] if "fps" in feature_attributes else None
