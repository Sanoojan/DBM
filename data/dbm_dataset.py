import os
import sys

import numpy as np
from numpy.dtypes import StringDType
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../hail-datasets/hail_datasets/datasets/ddd_2024"))
import ddd_2024_dataset
from ddd_2024_chunk import Chunk, CollatedChunk
from ddd_2024_dataset import FoldConfig
from ddd_2024_features.base_feature import BaseCollatedFeature, BaseFeature
from ddd_2024_features.dictionary_array_feature import (
    CollatedDictionaryArrayFeature,
    DictionaryArrayFeature,
)
from ddd_2024_features.object_attributes_feature import (
    CollatedObjectAttributesFeature,
    ObjectAttributesFeature,
)
from ddd_2024_features.object_tracks_feature import (
    CollatedObjectTracksFeature,
    ObjectTracksFeature,
)
from ddd_2024_features.tabular_feature import CollatedTabularFeature, TabularFeature
from ddd_2024_features.video_feature import CollatedVideoFeature, VideoFeature
from ddd_2024_participant import CollatedParticipant, Participant
from ddd_2024_sample import CollatedSample, Sample
from ddd_2024_scenario import CollatedScenario, Scenario

sys.path.pop()


class DBM_Dataset_Config(ddd_2024_dataset.DDD2024DatasetConfig):
    def __init__(self):
        super().__init__()


class DBM_Dataset(ddd_2024_dataset.DDD2024Dataset, Dataset):
    def __init__(self, config, train_dataset=None):
        super().__init__(config, train_dataset)
