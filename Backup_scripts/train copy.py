#!/usr/bin/env python3
import os

import numpy as np
import torch

from models.model_factory import load_model
from utils.output_logger import OutputLogger
from utils.parse_args import parse_args
from utils.torch_utils import create_dataloader, set_random_seed
from utils.visualization import visualize_batch


def main():
    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device", device)

    # Parse args
    args = parse_args("train")
    hyper_params = vars(args)

    # Fix random seed for reproducability
    set_random_seed(args.random_seed)

    # Setup default names
    if args.group_name is None:
        args.group_name = "null_group"
    if args.exp_name is None:
        args.exp_name = "null_exp"

    # Create output dir
    if args.output_dir is None:
        args.output_dir = os.path.join("checkpoints", args.group_name, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize output logger
    logger = OutputLogger(args.wandb_project_name is not None, args.output_dir)
    if args.wandb_project_name is not None:
        print("Initializing Weights and Biases")
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            group=hyper_params["group_name"],
            name=hyper_params["exp_name"],
            tags=hyper_params["exp_tags"],
            config=hyper_params,
        )

    # Set up dataloaders
    dataloaders = {}
    include_objects = args.include_object_tracks
    print(f"Include object tracks: {include_objects}")
    dataloaders["train"] = create_dataloader(args, fold_name="train", include_objects=include_objects)
    dataloaders["val"] = create_dataloader(
        args, fold_name="val", include_objects=include_objects, train_dataset=dataloaders["train"].dataset
    )
    dataloaders["test"] = create_dataloader(
        args, fold_name="test", include_objects=include_objects, train_dataset=dataloaders["train"].dataset
    )

    # Set up model
    model = load_model(args, dataloaders["train"], device)

    # Loop through epochs
    best_val_stats = {}
    best_test_stats = {}
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        # Loop through fold dataloaders and get outputs
        fold_stats = {}
        model_outputs = {}
        for fold_name, dataloader in dataloaders.items():
            if fold_name != "train" and epoch % args.val_num_epochs != args.val_num_epochs - 1:
                continue
            model_outputs[fold_name] = []

            # Let the model know the current fold
            model.current_fold = fold_name

            # Loop through batches in epoch and get outputs
            total_batches = len(dataloader)
            for batch_idx, batch in enumerate(dataloader):
                print(f"{fold_name} batch {batch_idx+1} / {total_batches}")
                model_output = model.evaluate_and_update(batch, train=(fold_name == "train"), batch_idx=batch_idx)

                # Add additional metadata about each sample
                model_output["relative_path"] = batch.relative_path
                model_output["chunk_start"] = batch.chunk.start_epoch_ns
                model_output["chunk_end"] = batch.chunk.end_epoch_ns

                model_outputs[fold_name].append(model_output)
                if batch_idx == 0 and epoch % args.vis_num_epochs == args.vis_num_epochs - 1:
                    visualize_batch(model_output, batch, fold_name, epoch, batch_idx, logger)

            # Finalize epoch and get stats for fold
            fold_stats[fold_name] = {}
            epoch_stats = model.score(model_outputs[fold_name])
            for stat_name, stat in epoch_stats.items():
                fold_stat_name = f"{fold_name}_{stat_name}"
                print(f"{fold_stat_name}: {stat:.3f}")
                fold_stats[fold_name][fold_stat_name] = stat
            logger.output_values(fold_stats[fold_name], epoch)

        # Save last model checkpoint
        model.save(os.path.join(weights_dir, "last"))

        # Update best validation test stats and save checkpoints / outputs
        if epoch % args.val_num_epochs == args.val_num_epochs - 1 and "val" in fold_stats and "test" in fold_stats:
            for val_stat_name, val_stat in fold_stats["val"].items():
                if len(args.val_test_metrics) > 0 and val_stat_name not in args.val_test_metrics:
                    continue
                best_val_stat_name = f"best_{val_stat_name}"

                # Determine if best seen validation performance for metric
                if (
                    best_val_stat_name not in best_val_stats
                    or np.isnan(best_val_stats[best_val_stat_name])
                    or best_val_stats[best_val_stat_name] > val_stat
                ):
                    # Save val stat
                    print(f"NEW BEST {val_stat_name}: {val_stat:.3f}")
                    best_val_stats[best_val_stat_name] = val_stat

                    # Save test stats
                    for test_stat_name, test_stat in fold_stats["test"].items():
                        best_test_stat_name = f"{best_val_stat_name}-{test_stat_name}"
                        best_test_stats[best_test_stat_name] = test_stat

                    # Save model outputs for all folds
                    if args.save_best_outputs:
                        logger.save_best_model_outputs(best_val_stat_name, model_outputs)

                    # Save model checkpoint for best val results
                    model.save(os.path.join(weights_dir, best_val_stat_name))

            # Log the best val and test stats
            logger.output_values(best_val_stats, epoch)
            logger.output_values(best_test_stats, epoch)

    # Finish training
    logger.finalize()


if __name__ == "__main__":
    main()
