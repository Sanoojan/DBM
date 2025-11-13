import numpy as np
from scipy import stats
import warnings


def calculate_aggregate(aggregate_data, aggregate_func, fps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if aggregate_func == "mean":
            return np.nanmean(aggregate_data)
        elif aggregate_func == "abs_mean":
            return np.nanmean(np.abs(aggregate_data))
        elif aggregate_func == "pos_mean":
            filtered_data = aggregate_data[aggregate_data > 0]
            return 0. if len(filtered_data)==0 else np.nanmean(filtered_data)
        elif aggregate_func == "neg_mean":
            filtered_data = aggregate_data[aggregate_data < 0]
            return 0. if len(filtered_data)==0 else -np.nanmean(filtered_data)
        elif aggregate_func == "sum_rate":
            chunk_duration_min = np.sum(np.logical_not(np.isnan(aggregate_data))) / (fps * 60)
            return np.nansum(aggregate_data) / chunk_duration_min
        elif aggregate_func == "std":
            return np.nanstd(aggregate_data)
        elif aggregate_func == "change_rate":
            chunk_duration_min = np.sum(np.logical_not(np.isnan(aggregate_data))) / (fps * 60)
            filtered_data = aggregate_data[np.logical_not(np.isnan(aggregate_data))]
            return np.nansum(np.diff(filtered_data == 1)) / chunk_duration_min
        elif aggregate_func == "skewness": 
            return stats.skew(aggregate_data, nan_policy="omit")
        elif aggregate_func == "kurtosis":
            return stats.kurtosis(aggregate_data, nan_policy="omit")
        elif aggregate_func == "cid_ce":
            normalized_data = (aggregate_data - np.nanmean(aggregate_data)) / np.nanstd(aggregate_data)
            return np.sqrt(np.nansum(np.square(np.diff(normalized_data))))
        else:
            raise Exception("Unknown aggregate function " + aggregate_func)
