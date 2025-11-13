import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.carla import get_carla_object_current_yaw
from utils.math import rotate_2d
from utils.video import get_segmentation_color, get_segmentation_most_salient

cmap = matplotlib.colormaps["winter"]


def plot_tracks(tracks, current_resample_ns, is_ego=True, past_duration_s=3.0, offset_xy=None, rotate_yaw=None):
    # Get tracks
    start_ns = current_resample_ns - int(past_duration_s * 1e9)
    tracks_subset = tracks.loc[start_ns:current_resample_ns]
    xy = tracks_subset[["carla_objects_pose_x", "carla_objects_pose_y"]].to_numpy()
    if xy.shape[0] == 0:
        return

    # Normalize
    if offset_xy is not None:
        xy -= offset_xy
    if rotate_yaw is not None:
        c, s = np.cos(rotate_yaw), np.sin(rotate_yaw)
        rotate_R = np.array(((c, -s), (s, c)))
        xy = xy @ rotate_R

    # Determine style
    if is_ego:
        color = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        # ado
        # Base color on final distance from ego
        # dist = np.linalg.norm(xy[-1, :])
        # c_index = np.clip(dist / 50.0, 0.0, 1.0)
        # color = np.array(cmap(c_index))
        color = np.array([0, 0, 142, 255]) / 255.0

    alpha_color = color.copy()
    alpha_color[-1] = 0.5
    plt.plot(xy[:, 0], xy[:, 1], color=alpha_color)
    if tracks_subset.index[-1] == current_resample_ns:
        actor_yaw = get_carla_object_current_yaw(tracks_subset)
        if rotate_yaw is not None:
            actor_yaw -= rotate_yaw
        plt.scatter(xy[-1, 0], xy[-1, 1], color=color, marker=(3, 0, np.rad2deg(actor_yaw)), edgecolors="black")


def draw_gaze_ray(gaze_in_xy):
    gaze_in_xy = np.array([[0.0, 0.0], gaze_in_xy])
    left_ray = rotate_2d(gaze_in_xy, 5.0)
    left_ray[1] += 1.5
    right_ray = rotate_2d(gaze_in_xy, -5.0)
    right_ray[1] += 1.5
    tri_points = np.array([left_ray[0], left_ray[1], right_ray[1]])

    tri_yellow = np.array([1.0, 1.0, 0.0, 0.25])
    tri = plt.Polygon(tri_points, color=tri_yellow)
    plt.gca().add_patch(tri)
    plot_yellow = np.array([1.0, 0.8, 0.0, 0.75])
    plt.plot(left_ray[:, 0], left_ray[:, 1], color=plot_yellow, linestyle="--")
    plt.plot(right_ray[:, 0], right_ray[:, 1], color=plot_yellow, linestyle="--")


def get_trajectory_image(sample, current_resample_ns, past_duration_s=3.0, plot_size=[640, 480]):
    my_dpi = 100
    fig = plt.figure(figsize=(plot_size[0] / my_dpi, plot_size[1] / my_dpi), dpi=my_dpi)

    # Determine ego normalization parameters
    start_ns = current_resample_ns - int(past_duration_s * 1e9)
    ego_tracks_subset = sample.features["ego_tracks"].table.loc[start_ns:current_resample_ns]
    ego_tracks_subset.columns = ego_tracks_subset.columns.str.removeprefix("ego_tracks-")

    offset_xy = ego_tracks_subset[["carla_objects_pose_x", "carla_objects_pose_y"]].iloc[-1].to_numpy()
    rotate_yaw = get_carla_object_current_yaw(ego_tracks_subset)

    # Plot ado trajectories
    for ado_id, ado_tracks in sample.features["ado_tracks"].tracks.items():
        plot_tracks(ado_tracks, current_resample_ns, False, past_duration_s, offset_xy, rotate_yaw)

    # Plot ego trajectory
    plot_tracks(ego_tracks_subset, current_resample_ns, True, past_duration_s, offset_xy, rotate_yaw)

    # Add ego gaze
    tabular_subset = sample.features["tobii"].table.loc[start_ns:current_resample_ns]
    tobii_name = "tobii_left_eye_gaze_pt_in_display_x"
    gaze_in_screen = tabular_subset[tobii_name].iloc[-1]
    fov = np.deg2rad(90.0)
    gaze_h_angle = (gaze_in_screen * fov) - (fov / 2.0)
    gaze_in_xy = np.array([np.sin(gaze_h_angle), np.cos(gaze_h_angle)])
    gaze_in_xy *= 100.0
    draw_gaze_ray(gaze_in_xy)

    # Set axis limits
    ax = plt.gca()
    ax.set_xlim([-50.0, 50.0])
    ax.set_ylim([-20.0, 130.0])
    ax.set_aspect("equal", adjustable="datalim")
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Draw and get the image data
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
    plt.close()

    return data
