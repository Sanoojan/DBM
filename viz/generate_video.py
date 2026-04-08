#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np

from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config
from viz.trajectory_plot import get_trajectory_image
from viz.video_with_gaze import get_video_with_gaze_frame


def generate_sample(sample, args):
    # Determine export type
    _, file_extension = os.path.splitext(args.output)
    if file_extension.lower() == ".png":
        export_video = False
    elif file_extension.lower() == ".mp4":
        export_video = True
    else:
        raise Exception("Unsupported file extension: " + file_extension)

    # Set FPS for video
    fps = 10

    # Get resampling times
    resample_times = sample.chunk.resample_times
    past_duration_s = 3.0
    start_ns = resample_times[0] + int(past_duration_s * 1e9)
    stop_ns = resample_times[-1]
    if export_video:
        stop_ns = resample_times[-1]
    else:
        stop_ns = start_ns

    # Loop through each time and create the frame
    video_writer = None
    for current_time in range(start_ns, stop_ns + 1, int(1e9) // fps):
        image_data = []

        # Get the videos for the frame
        for video_name in args.videos:
            video_frame = get_video_with_gaze_frame(sample, video_name, current_time, args)
            image_data.append(video_frame)

        # Get the bird's eye image for the frame
        if not args.no_trajectory:
            # Determine resolution to use
            if len(image_data) > 0:
                plot_size = [image_data[0].shape[1], image_data[0].shape[0]]
            else:
                plot_size = [960, 400]

            # Get the trajectory image
            image_data.append(get_trajectory_image(sample, current_time, past_duration_s, plot_size))

        # Combine all images
        columns = int(np.ceil(len(image_data) / args.rows))
        max_x = np.max([image.shape[1] for image in image_data])
        max_y = np.max([image.shape[0] for image in image_data])
        combined_frame = np.zeros((max_y * args.rows, max_x * columns, 3), dtype=np.uint8)
        image_it = 0
        for row_index in range(args.rows):
            for column_index in range(columns):
                if image_it >= len(image_data):
                    break
                x_left = column_index * max_x
                x_right = x_left + max_x
                y_top = row_index * max_y
                y_bottom = y_top + max_y
                combined_frame[y_top:y_bottom, x_left:x_right, :] = image_data[image_it][:, :, :3]
                image_it += 1

        # Export the frame
        if export_video:
            # Open the video writer
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    args.output, fourcc, fps, (combined_frame.shape[1], combined_frame.shape[0])
                )

            # Write the frame
            video_writer.write(combined_frame)
        else:
            # Write the image
            cv2.imwrite(args.output, combined_frame)

    # Close the video
    if video_writer is not None:
        video_writer.release()


def main():
    # Set up the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--no-gaze", action="store_true")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=[],
        help="The list of video modalities to include (cam_front, cam_depth, cam_seg, cam_face)",
    )
    parser.add_argument("--rows", type=int, default=1, help="The number of rows when concatenating videos")
    parser.add_argument("--scale", type=float, default=0.5, help="The scale to use for resizing videos")
    parser.add_argument("--filter", type=str, default="driving/[0-9].*", help="The filter to use to select scenarios")
    parser.add_argument("--duration", type=float, default=28.0, help="The duration of the chunk for video generation")
    parser.add_argument(
        "--end_offset", type=float, default=0.0, help="The offset from the end of the chunk for video generation"
    )
    parser.add_argument("--sample", type=int, default=0, help="The index of the sample to use")
    parser.add_argument(
        "--use_participants", nargs="+", default=None, help="The list of subjects to include (uses all by default)"
    )
    args = parser.parse_args()

    # Load the hazard dataset
    config = DBM_Dataset_Config()

    config.base_path = "dataset/Vehicle/No-Video/"
    config.index_relative_path = "Resampled_previous_10"

    config.features = {
        "tobii": {
            "relative_path": "Resampled_previous_10",
            "file_name": "experiment_tobii_frame",
            "type": "tabular",
            "fps": 10,
        }
    }

    if not args.no_trajectory:
        config.features["ego_tracks"] = {
            "relative_path": "Resampled_previous_10_derived",
            "type": "tabular",
            "fps": 10,
        }
        config.features["ado_tracks"] = {
            "relative_path": "Resampled_previous_10_derived",
            "type": "object_tracks",
            "fps": 10,
        }

    if "cam_front" in args.videos:
        config.features["cam_front"] = {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (int(400 * args.scale), int(960 * args.scale), 3),
            "interpolation": cv2.INTER_AREA,
            "skip_frames": True,
        }
    if "cam_depth" in args.videos:
        config.features["cam_depth"] = {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (int(400 * args.scale), int(960 * args.scale), 1),
            "interpolation": cv2.INTER_AREA,
            "video_type": "depth",
            "skip_frames": True,
        }
    if "cam_seg" in args.videos:
        config.features["cam_seg"] = {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (int(400 * args.scale), int(960 * args.scale), 1),
            "interpolation": cv2.INTER_NEAREST,
            "video_type": "segmentation",
            "skip_frames": True,
        }
    if "cam_face" in args.videos:
        config.features["cam_face"] = {
            "relative_path": "Resampled_previous_10",
            "type": "video",
            "fps": 10,
            "resolution": (int(960 * args.scale), int(960 * args.scale), 3),
            "interpolation": cv2.INTER_AREA,
            "skip_frames": True,
        }

    config.scenario_name_filter = args.filter  # "driving/8a-.*"  # "driving/[0-9].*"
    config.use_participants = args.use_participants

    config.chunk_strategy = "end"
    config.chunk_end_offset = args.end_offset
    config.chunk_duration = args.duration
    config.chunks_per_scenario = 1
    dataset = DBM_Dataset(config)
    print(f"Dataset contains {len(dataset)} samples")

    # Choose a random sample to visualize
    sample_number = args.sample
    sample = dataset[sample_number]
    print(sample.relative_path)
    generate_sample(sample, args)
    sample.close()


if __name__ == "__main__":
    main()
