#!/usr/bin/env python3
import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation  # or use PillowWriter for .gif
from matplotlib.cm import get_cmap


def visualize_batch(model_output, batch, fold_name, epoch, batch_idx, logger):
    # generate 2x2 plot of ego / tobii / ado tracks and save as wandb image
    for b in range(min(2, batch.features["ego_tracks"].table.shape[0])):
        rows = 2
        cols = 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

        # Plot the ego signals
        plt_idx = 0
        ax = axes.flat[plt_idx]
        for f in batch.features["ego_tracks"].table_columns:
            delta_time = np.array(range(batch.features["ego_tracks"].table.shape[1])) / batch.features["ego_tracks"].fps
            f_idx = batch.features["ego_tracks"].table_columns.index(f)
            ax.plot(delta_time, batch.features["ego_tracks"].table[b, :, f_idx], "b")
        ax.set_title(f"{fold_name} Ep {epoch} Bat {batch_idx} - ego signals")

        # Plot the ego position
        plt_idx += 1
        ax = axes.flat[plt_idx]
        delta_time = np.array(range(batch.features["ego_tracks"].table.shape[1])) / batch.features["ego_tracks"].fps
        x_idx = batch.features["ego_tracks"].table_columns.index("carla_objects_pose_x")
        y_idx = batch.features["ego_tracks"].table_columns.index("carla_objects_pose_y")
        ax.plot(batch.features["ego_tracks"].table[b, :, x_idx], batch.features["ego_tracks"].table[b, :, y_idx], "b")
        ax.plot(batch.features["ego_tracks"].table[b, 0, x_idx], batch.features["ego_tracks"].table[b, 0, y_idx], "bx")
        ax.set_title(f"Ego xy pose ({delta_time[-1]-delta_time[0]}s)")

        # Plot the Tobii signals
        plt_idx += 1
        ax = axes.flat[plt_idx]
        for f in batch.features["experiment_tobii_frame"].table_columns:
            delta_time = (
                np.array(range(batch.features["experiment_tobii_frame"].table.shape[1]))
                / batch.features["experiment_tobii_frame"].fps
            )
            f_idx = batch.features["experiment_tobii_frame"].table_columns.index(f)
            ax.plot(delta_time, batch.features["experiment_tobii_frame"].table[b, :, f_idx], "r")
        ax.set_title(f"Tobii signals")

        # Plot the ado + ego positions
        # verify -- for some reason the carla objects for ego and ados seem to be very different (may be due to the long history)
        plt_idx += 1
        ax = axes.flat[plt_idx]
        if "object_tracks" in batch.features:
            for a in range(batch.features["object_tracks"].tracks.shape[1]):
                x_idx = batch.features["object_tracks"].tracks_columns.index("carla_objects_pose_x")
                y_idx = batch.features["object_tracks"].tracks_columns.index("carla_objects_pose_y")
                carla_ado_x = batch.features["object_tracks"].tracks[b, a, :, x_idx]
                carla_ado_y = batch.features["object_tracks"].tracks[b, a, :, y_idx]
                valid_idx = np.logical_and(carla_ado_x != 0, carla_ado_y != 0)
                ax.plot(carla_ado_x[valid_idx], carla_ado_y[valid_idx], "b")
                ax.plot(carla_ado_x[valid_idx[0]], carla_ado_y[valid_idx[0]], "bx")
            ax.axis("equal")
        x_idx = batch.features["ego_tracks"].table_columns.index("carla_objects_pose_x")
        y_idx = batch.features["ego_tracks"].table_columns.index("carla_objects_pose_y")
        ax.plot(batch.features["ego_tracks"].table[b, :, x_idx], batch.features["ego_tracks"].table[b, :, y_idx], "r-.")
        ax.plot(batch.features["ego_tracks"].table[b, 0, x_idx], batch.features["ego_tracks"].table[b, 0, y_idx], "rx")
        ax.set_title(f"Ado+ego position")

        # Log the plot to wandb
        images = []
        images.append(wandb.Image(fig))
        values = {f"BEV_{fold_name}_epoch_{epoch}_batch_{batch_idx},sample{b}": images}
        logger.output_values(values, epoch)

        plt.close(fig)

