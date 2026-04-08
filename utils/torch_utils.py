import math
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config, FoldConfig
from features.feature_sets import get_feature_set_config


def set_random_seed(seed):
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ICanDetach(ABC):
    @abstractmethod
    def detach(self):
        raise NotImplementedError()


def recursive_detach(data):
    if data is None:
        return None
    elif torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, ICanDetach):
        return data.detach()
    elif isinstance(data, dict):
        for key, value in data.items():
            data[key] = recursive_detach(data[key])
        return data
    else:
        return data


def init_weights(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def apply_yaw_rotations(xy, yaw):
    """
    Rotate a matrix of 2d points by yaws
    Args:
        xy (torch.Tensor): trajectory position / velocity data of shape (..., 2)
        yaw (torch.Tensor): yaw data in radians should match above shape except for final dimension (..., 1)
    """
    in_shape = xy.shape
    assert in_shape[-1] == 2
    rotate_R = xy.new_empty(in_shape + (2,))
    rotate_R[..., 0, 0] = torch.cos(yaw[..., 0])
    rotate_R[..., 0, 1] = -torch.sin(yaw[..., 0])
    rotate_R[..., 1, 0] = -rotate_R[..., 0, 1]
    rotate_R[..., 1, 1] = rotate_R[..., 0, 0]
    xy = xy[..., None, :] @ rotate_R
    return xy[..., 0, :]


def apply_yaw_cs_rotations(xy, yaw_cs, reverse=False):
    """
    Rotate a matrix of 2d points by yaw sines and cosines
    Args:
        xy (torch.Tensor): trajectory position / velocity data of shape (..., 2)
        yaw (torch.Tensor): yaw data in radians should match above shape except for final dimension (..., 1)
    """
    in_shape = xy.shape
    assert in_shape[-1] == 2
    sin_mult = -1.0 if reverse else 1.0
    rotate_R = xy.new_empty(in_shape + (2,))
    rotate_R[..., 0, 0] = yaw_cs[..., 0]
    rotate_R[..., 0, 1] = -yaw_cs[..., 1] * sin_mult
    rotate_R[..., 1, 0] = -rotate_R[..., 0, 1]
    rotate_R[..., 1, 1] = rotate_R[..., 0, 0]
    xy = xy[..., None, :] @ rotate_R
    return xy[..., 0, :]


def safe_atan2(sin_vals, cos_vals, eps=1e-7):
    need_eps = torch.abs(cos_vals) < eps
    cos_vals = (need_eps * eps) + (~need_eps * cos_vals)
    return torch.atan2(sin_vals, cos_vals)


def lerp_values(start=None, end=None, weight=None, is_angle=False):
    """
    Linearly interpolate between two tensors using a weight. All tensors must be broadcastable.
    Args:
        start (torch.Tensor): The tensor closer to 0. If None, the end tensor is returned.
        end (torch.Tensor): The tensor closer to 1. If None, the start tensor is returned. If both are None, None is returned.
        weight (torch.Tensor): The tensor defining the weights between 0 and 1 for selecting between start and end. If None, start is returned.
            If weight tensor is not floating point, select start if False and end if True.
        is_angle (bool): Whether or not to treat the inputs as angles. Default is False. Doesn't matter if weight is boolean.
    """
    if weight is None:
        return start
    if start is None:
        return end
    if end is None:
        return start
    if not torch.is_floating_point(weight):
        return (~weight * start) + (weight * end)
    if not is_angle:
        return torch.lerp(start, end, weight)
    else:
        start_cs = torch.stack([torch.cos(start), torch.sin(start)], dim=-1)
        end_cs = torch.stack([torch.cos(end), torch.sin(end)], dim=-1)
        merged_vec_cs = torch.lerp(start_cs, end_cs, weight[..., None])
        return torch.atan2(merged_vec_cs[..., 1], merged_vec_cs[..., 0])


# https://en.m.wikipedia.org/wiki/Smoothstep
def smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def create_dataloader(args, fold_name="train", include_objects=False, train_dataset=None):
    """
    Creates a DataLoader for the DBM dataset.
    Args:
        args: An object containing the necessary arguments, including:
            - test_split_index (int): Index for the test split.
            - batch_size (int): Batch size for the DataLoader.
        fold_name (str, optional): Name of the fold to be used. Defaults to "train".
        include_objects (bool, optional): Whether to include object data in the dataset. Defaults to False.
    Returns:
        DataLoader: A DataLoader object for the specified dataset configuration.
    Raises:
        ValueError: If any of the required arguments are missing or invalid.
    Example:
        dataloader = create_dataloader(args, fold_name="validation", include_objects=True)
    """
    config = DBM_Dataset_Config()

    config.base_path = os.path.expanduser(os.path.expandvars("dataset/Vehicle/No-Video/"))
    config.index_relative_path = "Resampled_previous_10"

    aggregate_model = args.model in [
        "RandomForest",
        "GradientBoosting",
        "ShallowNet",
        "SVM",
        "LogisticRegression",
        "MaskedAutoencoder",
    ]
    normalize_temporal = "none" if aggregate_model else args.feature_normalization
    normalize_aggregate = args.feature_normalization if aggregate_model else "none"
    config.features = get_feature_set_config(args.feature_set, normalize_temporal, normalize_aggregate, include_objects)

    config.experiment_version = args.experiment_version
    config.remove_participants = ["P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"]

    config.test_split_index = args.test_split_index
    config.fold_configs = [FoldConfig(name="fixed_fold", num_splits=5, split_train_val=True)]
    # config.fold_configs = [FoldConfig(name="participant_name", num_splits=5, split_train_val=True)]
    config.fold_name = fold_name
    config.downsample = args.downsample
    config.exclude_cd = args.exclude_cd
    config.exclude_intoxicated = args.exclude_intoxicated
    config.train_downsample_intox = args.train_downsample_intox
    config.exclude_baseline = args.exclude_baseline
    config.random_seed = args.random_seed

    config.chunk_end_offset = 30.0
    config.chunk_duration = args.chunk_duration
    if fold_name == "train":
        config.scenario_name_filter = "driving/[0-9].*"  # "driving/6e.*"
        config.chunk_start_offset = 30.0
        config.chunk_strategy = "random"  # args.train_chunk_strategy
        config.chunks_per_scenario = None  # args.train_chunks_per_scenario
        config.chunk_stride = config.chunk_duration
    else:
        config.chunk_end_offset = 30.0
        config.scenario_name_filter = "driving/[0-9].*"  # "driving/6e.*"
        config.chunk_start_offset = 30.0
        config.chunk_strategy = "end"
        config.chunks_per_scenario = 4
        config.chunk_stride = config.chunk_duration
    config.cache_chunks = fold_name != "train"

    dataset = DBM_Dataset(config, train_dataset)
    num_samples = len(dataset)
    print(f"Created {fold_name} dataset of {num_samples} samples")

    if fold_name == "train":
        if args.dataset_size_multiplier is None:
            sampler = torch.utils.data.RandomSampler(dataset, num_samples=num_samples, replacement=False)
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset, num_samples=math.ceil(num_samples * args.dataset_size_multiplier), replacement=True
            )
        # import IPython;IPython.embed(header="Sample")
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=dataset.collate,
        num_workers=args.num_dataset_workers,
    )

    return dataloader
