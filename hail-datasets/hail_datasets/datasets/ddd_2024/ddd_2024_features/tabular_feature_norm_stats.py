import numpy as np


# Class to keep track of single normalization feature set
# https://www.johndcook.com/blog/standard_deviation/
class TabularFeatureNormStat():
    def __init__(self):
        self.m_n = None
        self.finalized = False

    def push(self, data):
        if self.m_n is None:
            self.m_n = np.zeros((data.shape[1],), dtype=int)
            self.m_oldM = np.zeros((data.shape[1],), dtype=float)
            self.m_newM = np.zeros((data.shape[1],), dtype=float)
            self.m_oldS = np.zeros((data.shape[1],), dtype=float)
            self.m_newS = np.zeros((data.shape[1],), dtype=float)
        for i in range(data.shape[0]):
            row = data[i]
            valid = np.logical_not(np.isnan(row))
            self.m_n += valid

            # Override on first valid iteration
            valid_and_first = np.logical_and(valid, self.m_n==1)
            self.m_oldM[valid_and_first] = row[valid_and_first]
            self.m_newM[valid_and_first] = row[valid_and_first]

            # Calculate running stats
            valid_and_later = np.logical_and(valid, self.m_n>1)
            self.m_newM[valid_and_later] = self.m_oldM[valid_and_later] + (row[valid_and_later] - self.m_oldM[valid_and_later]) / self.m_n[valid_and_later]
            self.m_newS[valid_and_later] = self.m_oldS[valid_and_later] + (row[valid_and_later] - self.m_oldM[valid_and_later]) * (row[valid_and_later] - self.m_newM[valid_and_later])
            self.m_oldM[valid_and_later] = self.m_newM[valid_and_later]
            self.m_oldS[valid_and_later] = self.m_newS[valid_and_later]

    def num(self):
        return self.m_n

    def mean(self):
        return self.m_newM

    def var(self):
        return self.m_newS / (self.m_n - 1)

    def std(self):
        return np.sqrt(self.var())
    
    def finalize(self):
        self.finalized = True

    def is_finalized(self):
        return self.finalized

# Class to keep track of tabular normalization stats
class TabularFeatureNormStats():
    def __init__(self):
        self.running_totals = {}

    def apply(self, table, participant_name, feature_name, norm_method):
        # Determine how to group normalization parameters
        if norm_method == "population":
            group_name = "population"
        elif norm_method == "subject":
            group_name = participant_name
        elif norm_method == "none":
            return table
        else:
            raise Exception(f"Unknown normalization method {norm_method}")

        # Create the norm stats for the group and feature
        if group_name not in self.running_totals:
            self.running_totals[group_name] = {}
        if feature_name not in self.running_totals[group_name]:
            self.running_totals[group_name][feature_name] = TabularFeatureNormStat()
        
        # Normalize the feature if parameters are finalized, otherwise update them
        if self.running_totals[group_name][feature_name].is_finalized():
            return (table - self.running_totals[group_name][feature_name].mean()) / self.running_totals[group_name][feature_name].std()
        else:
            self.running_totals[group_name][feature_name].push(table.to_numpy())
            return table

    def finalize(self):
        for group_name, group_dict in self.running_totals.items():
            for feature_name, feature_stats in group_dict.items():
                feature_stats.finalize()
