import numpy as np

from data.dbm_dataset import CollatedTabularFeature


class AggregateFeatures:
    def __init__(self, model):
        self.model = model

    # Gathers the tabular features for the random forest
    def collect_features(self, batch):
        features = []
        for _, feature in batch.features.items():
            if not isinstance(feature, CollatedTabularFeature):
                continue
            if not feature.has_aggregate:
                continue
            X = feature.aggregate[:, [x not in self.model.args.model_ignore_columns for x in feature.aggregate_columns]]
            features.append(X)
        return np.concatenate(features, axis=-1)
