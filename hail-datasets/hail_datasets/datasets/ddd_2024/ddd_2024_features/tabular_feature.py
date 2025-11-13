import pandas as pd
import numpy as np

from ddd_2024_features.base_feature import BaseFeature, BaseCollatedFeature
from ddd_2024_features.common import get_matrix_from_tabular, format_columns
from ddd_2024_features.tabular_feature_norm_stats import TabularFeatureNormStat, TabularFeatureNormStats
from ddd_2024_features.tabular_aggregate_functions import calculate_aggregate


def format_aggregate_name(aggregate_dict, aggregate_func):
    if "name" in aggregate_dict:
        return aggregate_dict["name"] + "__" + aggregate_func
    elif "column" in aggregate_dict:
        return aggregate_dict["column"] + "__" + aggregate_func
    else:
        raise Exception("Must specify output aggregate name when using multiple columns")

def format_attributes(feature_attributes):

    if "aggregate" in feature_attributes:
        # Initialize the column set
        if "columns" not in feature_attributes:
            feature_attributes["columns"] = set()
        else:
            feature_attributes["columns"] = set(feature_attributes["columns"])

        # Find the new columns
        for aggregate_dict in feature_attributes["aggregate"]:
            if "column" in aggregate_dict:
                feature_attributes["columns"].add(aggregate_dict["column"])
            elif "columns" in aggregate_dict:
                for column in aggregate_dict["columns"]:
                    feature_attributes["columns"].add(column)

        # Convert back to a list
        feature_attributes["columns"] = list(feature_attributes["columns"])
    return feature_attributes

# Tabular - data variable in length over time but constant in number of columns
class TabularFeature(BaseFeature):
    def __init__(self, feature_name, feature_attributes, full_scenario_path, sample):
        super().__init__(feature_name, feature_attributes)\
        
        # Format the attributes
        feature_attributes = format_attributes(feature_attributes)

        # Load the csv data
        csv_path = self.get_feature_path(full_scenario_path, feature_attributes["file_name"] + ".csv")
        csv_columns = feature_attributes["columns"] if "columns" in feature_attributes else None
        self.table = self.load_dataframe(csv_path, index="resampled_epoch_ns", column_list=csv_columns) 

        # Filter by chunk times - copy if not in cache
        self.table = sample.chunk.filter_dataframe(self.table)
        if not self.cache_features:
            self.table = self.table.copy()

        # Perform normalization
        if "normalize" in feature_attributes:
            self.table = sample.dataset.tabular_norm_stats.apply(self.table, sample.participant.name, feature_name, feature_attributes["normalize"])

        # Aggregate functions
        self.has_aggregate = ("aggregate" in feature_attributes)
        if "aggregate" in feature_attributes:
            self.aggregate = {}
            for aggregate_dict in feature_attributes["aggregate"]:
                # Get aggregate data
                
                if "column" in aggregate_dict:
                    column_name = aggregate_dict["column"]
                    if column_name not in self.table.columns:
                        continue
                    aggregate_data = self.table[column_name].to_numpy()[:,None]
                elif "columns" in aggregate_dict:
                    if not pd.Series(aggregate_dict["columns"]).isin(self.table.columns).all():
                        continue
                    aggregate_data = self.table[aggregate_dict["columns"]].to_numpy()
                else:
                    raise Exception("Need to specify aggregate source column or columns")
                
                # If only single function given, populate list
                aggregate_funcs = aggregate_dict["functions"] if "functions" in aggregate_dict else [aggregate_dict["function"],]

                # Calculate aggregates
                for aggregate_func in aggregate_funcs:
                    out_name = format_aggregate_name(aggregate_dict, aggregate_func)
                    aggregate_sum = 0.
                    aggregate_weight_total = 0
                    for i in range(aggregate_data.shape[1]):
                        W = np.sum(np.logical_not(np.isnan(aggregate_data[:,i])))
                        aggregate_sum += W * calculate_aggregate(aggregate_data[:,i], aggregate_func, self.fps)
                        aggregate_weight_total += W
                    self.aggregate[out_name] = aggregate_sum / aggregate_weight_total if aggregate_weight_total != 0 else np.nan                        
                
            # Convert to pandas
            self.aggregate = pd.DataFrame([self.aggregate,])

            # Perform normalization
            if "normalize_aggregate" in feature_attributes:
                self.aggregate = sample.dataset.tabular_norm_stats.apply(self.aggregate, sample.participant.name, feature_name + "-agg", feature_attributes["normalize_aggregate"])


class CollatedTabularFeature(BaseCollatedFeature):
    # Populates the following:
    # self.table (batch_size, num_frames, num_columns)
    # self.table_valid (batch_size, num_frames, num_columns)
    # self.table_columns
    
    def __init__(self, feature_name, feature_attributes, list_of_features, collated_sample):
        super().__init__(feature_name, feature_attributes)
        
        # Format the attributes
        feature_attributes = format_attributes(feature_attributes)

        batch_size = collated_sample.batch_size

        if "keep_temporal" not in feature_attributes or feature_attributes["keep_temporal"]:
            fps = collated_sample.config.features[feature_name]["fps"]
            num_frames = collated_sample.feature_num_frames[fps]
            tabular_columns = format_columns(feature_attributes)

            self.table = np.zeros((batch_size, num_frames, len(tabular_columns)))
            self.table_valid = np.zeros((batch_size, num_frames, len(tabular_columns)))
            self.table_columns = tabular_columns
            for sample_on, feature_data in enumerate(list_of_features):
                if feature_data is not None:
                    tabular_data = get_matrix_from_tabular(feature_data.table, tabular_columns)
                    self.table_valid[sample_on] = np.logical_not(np.isnan(tabular_data))
                    tabular_data[np.isnan(tabular_data)] = 0.0
                    self.table[sample_on] = tabular_data

        self.has_aggregate = ("aggregate" in feature_attributes)
        if "aggregate" in feature_attributes:
            self.aggregate_columns = []
            for aggregate_dict in feature_attributes["aggregate"]:
                aggregate_funcs = aggregate_dict["functions"] if "functions" in aggregate_dict else [aggregate_dict["function"],]
                for aggregate_func in aggregate_funcs:
                    out_name = format_aggregate_name(aggregate_dict, aggregate_func)
                    self.aggregate_columns.append(out_name)           
            self.aggregate = np.zeros((batch_size, len(self.aggregate_columns)))
            self.aggregate_valid = np.zeros((batch_size, len(self.aggregate_columns)))
            for sample_on, feature_data in enumerate(list_of_features):
                if feature_data is not None:
                    for column_idx, column_name in enumerate(self.aggregate_columns):
                        if column_name in feature_data.aggregate.columns:
                            self.aggregate_valid[sample_on,column_idx] = True
                            self.aggregate[sample_on,column_idx] = feature_data.aggregate[column_name][0]
