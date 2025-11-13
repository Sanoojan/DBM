import fnmatch
import glob
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm

# Set ggplot styles and update Matplotlib with them.
# sudo apt install ttf-mscorefonts-installer if missing
ggplot_styles = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "axes.edgecolor": "white",
    "axes.facecolor": "EBEBEB",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "grid.color": "white",
    "grid.linewidth": "1.2",
    "xtick.color": "555555",
    "xtick.major.bottom": True,
    "xtick.minor.bottom": False,
    "ytick.color": "555555",
    "ytick.major.left": True,
    "ytick.minor.left": False,
}
plt.rcParams.update(ggplot_styles)


def plot_log_time_delta(csv_path, window_size=30, ylim=None):
    """Plot the delta in "* log time" from a given csv file"""
    df = pd.read_csv(csv_path)
    log_time_column = [col for col in df.columns if col.endswith(" log time")]
    assert len(log_time_column) == 1, print(f"Found {len(log_time_column)} log time fields in csv")
    log_time_column = log_time_column[0]
    plt.figure(figsize=(15, 6))
    plt.plot(df[log_time_column].diff().rolling(window=window_size).mean().values)
    plt.title(f"{log_time_column} delta")
    if ylim:
        plt.ylim(ylim)
    plt.xlabel("Index")
    plt.ylabel("Delta time (s)")
    plt.grid(True)
    plt.show()


def plot_scenario_durations(scenario_durations, scenario_abbreviations):
    scenario_durations_median_adjusted = scenario_durations - scenario_durations.median()
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("coolwarm")
    vmin, vmax = -180, 180
    plt.imshow(scenario_durations_median_adjusted, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
    plt.xticks(
        ticks=np.arange(len(scenario_durations.columns)),
        labels=[scenario_abbreviations[_] for _ in scenario_durations.columns],
        rotation=90,
    )
    plt.yticks(ticks=np.arange(len(scenario_durations.index)), labels=scenario_durations.index)
    plt.colorbar(label="Scenario duration deviation from median (s)")
    plt.title("Scenario duration matrix")
    plt.grid(False)
    plt.show()


def plot_duration_per_participant(scenario_durations):
    df = scenario_durations.copy()
    driving = df[[col for col in df.columns if fnmatch.fnmatch(col, "*driving*/*-*/*")]].sum(axis=1).values / 60
    practice = df[[col for col in df.columns if fnmatch.fnmatch(col, "*practice*")]].sum(axis=1).values / 60
    calib = df["R1-tobii_calibration"].values / 60 + df["R2-tobii_calibration"].values / 60
    stationary = df[[col for col in df.columns if fnmatch.fnmatch(col, "R*stationary*")]].sum(axis=1).values / 60
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(df.index, stationary, label="stationary", color="skyblue")
    bars2 = ax.bar(df.index, practice, bottom=stationary, label="practice", color="peachpuff")
    bars3 = ax.bar(df.index, driving, bottom=practice + stationary, label="hazards", color="teal")
    bars4 = ax.bar(df.index, calib, bottom=driving + practice + stationary, label="calibration", color="salmon")
    plt.legend(["stationary", "practice", "hazards", "calibration"])
    plt.xticks(ticks=np.arange(len(df.index)), labels=df.index, rotation=90)
    plt.ylabel("Total scenario duration (min)")
    plt.show()


def plot_freqs(scenario_freq, msg, v):
    scenario_freq_median_adjusted = scenario_freq - scenario_freq.median()
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("coolwarm")
    plt.imshow(scenario_freq_median_adjusted, cmap=cmap, vmin=v[0], vmax=v[1], interpolation="none")
    plt.xticks(ticks=np.arange(len(scenario_freq.columns)), labels=scenario_freq.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(scenario_freq.index)), labels=scenario_freq.index)
    plt.colorbar(label="Signal frequency deviation from median (Hz)")
    plt.title(f"{msg} frequency matrix")
    plt.grid(False)
    plt.show()


def plot_all_ego_trajectories(participant_id, root_dir="/data/motion-simulator-logs/Processed/Clean/Participants"):
    actor_file_list = sorted(
        glob.glob(os.path.join(root_dir, participant_id, "*/driving/*/*/sim_bag/carla_actor_list.csv"))
    )
    object_file_list = sorted(
        glob.glob(os.path.join(root_dir, participant_id, "*/driving/*/*/sim_bag/carla_objects.csv"))
    )
    for a, o in zip(actor_file_list, object_file_list):
        df = pd.read_csv(a)
        hero_id = df[df["carla_actor_rolename"] == "hero"]["carla_actor_id"].values[0]
        # hero_id is mostly 1000204 but occasionally 1000202(?!)
        df = pd.read_csv(o)
        ego_df = df[df["carla_objects_id"] == hero_id]
        plt.plot(ego_df["carla_objects_pose_x"].values, ego_df["carla_objects_pose_y"].values)
    plt.show()


def plotly_ego_trajectory_by_scenario(
    scenario_name, html=False, downsample=1, root_dir="/data/motion-simulator-logs/Processed/Clean/Participants"
):
    actor_file_list = sorted(
        glob.glob(os.path.join(root_dir, "*/*/driving", scenario_name, "*/sim_bag/carla_actor_list.csv"))
    )
    object_file_list = sorted(
        glob.glob(os.path.join(root_dir, "*/*/driving", scenario_name, "*/sim_bag/carla_objects.csv"))
    )
    traces = []
    for a, o in tqdm(zip(actor_file_list, object_file_list), desc=f"Processing {len(actor_file_list)} files"):
        df = pd.read_csv(a)
        hero_id = df[df["carla_actor_rolename"] == "hero"]["carla_actor_id"].values[0]
        # hero_id is mostly 1000204 but occasionally 1000202(?!)
        df = pd.read_csv(o)
        ego_df = df[df["carla_objects_id"] == hero_id]
        traces.append(
            go.Scatter(
                x=ego_df["carla_objects_pose_x"].values[::downsample],
                y=ego_df["carla_objects_pose_y"].values[::downsample],
                mode="lines",
                name=o.split("/")[-7],
                hoverinfo="name",
            )
        )
    layout = go.Layout(
        title=f"Trajectories for {scenario_name}",
        hovermode="closest",
        width=1000,
        height=1000,
    )

    fig = go.Figure(data=traces, layout=layout)

    # set x and y axes to be equally spaced
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # Link the scale of x-axis to the y-axis
            scaleratio=1,  # Set the scale ratio between x and y
        ),
        yaxis=dict(
            scaleanchor="x",  # Link the scale of y-axis to the x-axis
            scaleratio=1,  # Set the scale ratio between y and x
        ),
    )
    if html:
        html_fig = pio.to_html(fig, full_html=False)
        return html_fig
    else:
        fig.show()


def plotly_df_by_row(df, title):
    traces = []
    for i in range(1, len(df.index)):
        traces.append(go.Scatter(x=df.columns, y=df.iloc[i, :], mode="lines", name=df.index[i], hoverinfo="name+y"))

    layout = go.Layout(
        title=title,
        hovermode="closest",
        width=1000,
        height=1000,
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def generate_html_report(fig_list, output_file="dataset_report.html"):
    with open(output_file, "w") as f:
        f.write('<html><head><meta charset="utf-8" /></head><body>\n')
        for html_fig in fig_list:
            f.write(html_fig)
        f.write("</body></html>")


def plot_string_hist(data, figsize=(5, 2), ylim=(0, 300)):
    # make a histogram, plt.hist looks ugly so do as bar chart instead
    from collections import Counter

    frequency = Counter(data)
    labels, counts = zip(*frequency.items())

    plt.figure(figsize=figsize)
    bars = plt.bar(labels, counts)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va="bottom", ha="center", fontsize=12)
    plt.ylim(ylim)
    plt.show()
