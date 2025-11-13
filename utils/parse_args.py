import argparse
import os

import numpy as np


def parse_args(mode):
    parser = argparse.ArgumentParser()

    model_parser = parser.add_argument_group("Model")
    model_parser.add_argument("model", type=str, help="RandomForest, ShallowNet")
    # model_parser.add_argument("--pretrained_weights", type=str, default=None, help="if specified starts from checkpoint model")
    # model_parser.add_argument("--model_channels", type=int, default=64, help="the number of channels to use in the model")
    model_parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    model_parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    model_parser.add_argument(
        "--model_ignore_columns",
        nargs="+",
        type=str,
        default=["carla_objects_pose_x", "carla_objects_pose_y"],
        help="column names for the model to ignore",
    )
    model_parser.add_argument(
        "--model_tasks",
        nargs="+",
        type=str,
        default=["cd", "intoxicated", "impaired", "impaired_intoxicated"],
        help="the output tasks to predict",
    )
    model_parser.add_argument(
        "--model_task_weights", nargs="+", type=float, default=None, help="the output task weights (defaults to equal)"
    )

    # How to output results - wandb / output dir
    output_parser = parser.add_argument_group("Experiment Output")
    output_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="all output will be saved in this directory (if None, wil automatically set from group and experiment names)",
    )
    output_parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="which wandb project to log output to (default: do not use wandb)",
    )
    output_parser.add_argument("--group_name", type=str, default=None, help="interpretable name for wandb group")
    output_parser.add_argument("--exp_name", type=str, default=None, help="interpretable name for wandb run")
    output_parser.add_argument("--exp_tags", nargs="+", type=str, default=None, help="wandb experiment tags")

    # Optimization and loss functions
    opt_parser = parser.add_argument_group("Optimization and Losses")
    opt_parser.add_argument("--loss", nargs="+", type=str, default=[], help="L1, MSE, NegPC, NegSNR, MVTL, NegMCC, IPR")
    opt_parser.add_argument(
        "--loss_weights", nargs="+", type=float, default=None, help="The relative weight for each loss"
    )
    opt_parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
    opt_parser.add_argument("--num_dataset_workers", type=int, default=0, help="number of workers for dataset loading")
    opt_parser.add_argument("--optimizer", type=str, default="adamw", help="Select optimizer: adamw, sgd")
    opt_parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    opt_parser.add_argument("--scheduler", type=str, default=None, help="Select scheduler: linear, exponential")
    opt_parser.add_argument(
        "--scheduler_params", nargs="+", type=float, default=None, help="Scheduler-specific parameters"
    )

    # Training and validation
    train_parser = parser.add_argument_group("Training and Validation")
    train_parser.add_argument(
        "--train_metric",
        nargs="+",
        type=str,
        default=[],
        help="The metric or list of metrics used to determine train performance.",
    )
    # train_parser.add_argument('--train_num_epochs', type=int, default=5, help='How many epochs to train between calculating training metrics.')
    train_parser.add_argument(
        "--val_metric",
        nargs="+",
        type=str,
        default=[],
        help="The metric or list of metrics used to determine validation performance (in addition to model_last).",
    )
    train_parser.add_argument(
        "--val_num_epochs", type=int, default=1, help="How many epochs to train between calculating validation metrics."
    )
    train_parser.add_argument(
        "--vis_num_epochs",
        type=int,
        default=1,
        help="How many epochs to train between generating visualization outputs.",
    )
    train_parser.add_argument(
        "--val_test_metrics",
        nargs="+",
        default=[
            "loss",
        ],
        help="Limit the validation metrics used to select test results to only a subset (if empty, include all)",
    )
    train_parser.add_argument(
        "--save_best_outputs",
        action="store_true",
        help="flag to save the model outputs when validation metrics are best",
    )

    # Dataset
    dataset_parser = parser.add_argument_group("Dataset")
    dataset_parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    dataset_parser.add_argument(
        "--include_object_tracks", type=bool, default=False, help="include object data in dataset"
    )
    dataset_parser.add_argument(
        "--dataset_size_multiplier", type=float, default=None, help="multiplier for dataset size"
    )
    dataset_parser.add_argument(
        "--train_chunk_strategy", type=str, default="start", help="chunk strategy to use for training"
    )
    dataset_parser.add_argument(
        "--train_chunks_per_scenario",
        type=int,
        default=None,
        help="the number of training chunks to sample from each scenario (if None, use variable number)",
    )
    dataset_parser.add_argument(
        "--downsample", type=float, default=None, help="percent to downsample the dataset (if None, no downsampling)"
    )
    dataset_parser.add_argument(
        "--experiment_version", type=int, default=None, help="experiment version data to include (if None, use both)"
    )
    dataset_parser.add_argument(
        "--exclude_cd",
        nargs="+",
        type=str,
        default=[],
        help="the types of CD to exclude (nback_task, statement_task, or no_task)",
    )
    dataset_parser.add_argument(
        "--exclude_intoxicated",
        type=bool,
        default=None,
        help="the type of intoxication to exclude (False, True, or None - means to exclude nothing)",
    )
    dataset_parser.add_argument(
        "--feature_normalization",
        type=str,
        default="population",
        help="the feature normalization to use (population*, subject, or none)",
    )
    dataset_parser.add_argument(
        "--feature_set", type=str, default="main", help="which set of features to use. Options: `main` or `old`"
    )
    dataset_parser.add_argument(
        "--train_downsample_intox",
        type=float,
        default=None,
        help="percent to downsample intoxicated data in train set (if None, no downsampling)",
    )
    dataset_parser.add_argument(
        "--exclude_baseline",
        action="store_true",
        help="flag to exclude purely baseline data from analysis",
    )
    dataset_parser.add_argument(
        "--chunk_duration",
        type=float,
        default=30.0,
        help="Duration of chunks to extract in seconds",
    )

    # Folds

    # Experiment initialization
    experiment_parser = parser.add_argument_group("Experiment")
    # experiment_parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during generation')
    experiment_parser.add_argument("--random_seed", type=int, default=0, help="random seed to use for experiment")
    experiment_parser.add_argument(
        "--test_split_index",
        type=int,
        default=0,
        help="which split index to use as test when using cross validation",
    )
    experiment_parser.add_argument(
        "--num_repeats", type=str, default=None, help="used to specify repeat runs in the experiment launcher"
    )
    experiment_parser.add_argument(
        "--test_split_index_range",
        type=str,
        default=None,
        help="used to specify a range of test_split_index runs in the experiment launcher",
    )

    # Verify arguments
    args = parser.parse_args()
    if args.model_task_weights is None:
        args.model_task_weights = [1.0] * len(args.model_tasks)
    weight_total = np.sum(args.model_task_weights)
    args.model_task_weights = [x / weight_total for x in args.model_task_weights]
    args.val_test_metrics = [f"val_{x}" for x in args.val_test_metrics]

    return args
