import numpy as np


def get_matrix_from_tabular(data, columns, resample_times=None):
    if resample_times is None:
        data = data.reindex(columns=columns)
    else:
        data = data.reindex(index=resample_times, columns=columns)
    return np.array(data.values, dtype=np.float32)

def format_columns(feature_attributes):
    if "columns" not in feature_attributes:
        feature_name = feature_attributes["name"]
        raise Exception(f"Must specify columns for feature {feature_name} when using collation.")
    column_list = feature_attributes["columns"]
    if not isinstance(column_list, list):
        column_list = [
            column_list,
        ]
    return column_list
