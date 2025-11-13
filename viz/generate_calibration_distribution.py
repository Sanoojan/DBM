#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config


def main():
    # Load the entire dataset
    config = DBM_Dataset_Config()
    config.scenario_name_filter = "tobii_calibration"
    dataset = DBM_Dataset(config)

    # Get the durations
    durations = dataset.scenario_index["duration_s"].to_numpy()
    durations /= 60.0

    # Set up the plot
    plot_size = [640, 480]
    my_dpi = 100
    fig = plt.figure(figsize=(plot_size[0] / my_dpi, plot_size[1] / my_dpi), dpi=my_dpi)

    # Plot the distribution
    plt.hist(durations, bins=16)

    # Label axes
    ax = plt.gca()
    ax.set_xlabel("Minutes for Calibration")
    ax.set_ylabel("Count")

    # Draw and get the image data
    fig.canvas.draw()
    fig.savefig("calibration_durations.png")
    plt.close()

    print(
        "Percent calibrations taking less than two minutes: {:.1f}%".format(
            100 * np.sum(durations <= 2.0) / len(durations)
        )
    )
    print(
        "Percent calibrations taking less than three minutes: {:.1f}%".format(
            100 * np.sum(durations <= 3.0) / len(durations)
        )
    )
    print(
        "Percent calibrations taking less than four minutes: {:.1f}%".format(
            100 * np.sum(durations <= 4.0) / len(durations)
        )
    )


if __name__ == "__main__":
    main()
