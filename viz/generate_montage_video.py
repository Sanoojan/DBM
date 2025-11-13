import argparse
import glob
import os
import random

import cv2
import numpy as np


def downsample_video(input_path, output_path, factor):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_width = width // factor
    new_height = height // factor

    # save space
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (new_width, new_height))
        out.write(frame_resized)

    cap.release()
    out.release()


def create_montage(video_paths, grid_size, fps, downsample_factor, output_path):
    rows, cols = grid_size
    if len(video_paths) != rows * cols:
        raise ValueError("Number of videos must match rows x cols")

    resized_paths = []
    for idx, path in enumerate(video_paths):
        tmp_path = f"tmp_resized_{idx}.mp4"
        print(f"Downsampling {idx+1} of {len(video_paths)}")
        downsample_video(path, tmp_path, downsample_factor)
        resized_paths.append(tmp_path)

    caps = [cv2.VideoCapture(p) for p in resized_paths]
    frame_list = []

    w, h = None, None
    while True:
        frames = []
        count = 0
        for ii, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # if video ends early, append black frame
                frame = np.zeros_like((w, h, 3))
                count += 1
            if not w:
                w = frame.shape[0]
            if not h:
                h = frame.shape[1]
            frames.append(frame)

        if count == len(resized_paths):
            break

        frame_list.append(frames)

    for cap in caps:
        cap.release()

    if not frame_list:
        raise ValueError("Missing video frames to create a montage")

    frame_height, frame_width = frame_list[0][0].shape[:2]
    montage_width = frame_width * cols
    montage_height = frame_height * rows

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (montage_width, montage_height))

    for frames in frame_list:
        montage_frame = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                frame = frames[i * cols + j]
                y_start = i * frame_height
                y_end = y_start + frame_height
                x_start = j * frame_width
                x_end = x_start + frame_width
                montage_frame[y_start:y_end, x_start:x_end] = frame
        out.write(montage_frame)

    out.release()

    for path in resized_paths:
        os.remove(path)


if __name__ == "__main__":
    """
    Example:
        python generate_montage_video.py \
            "/data/motion-simulator-logs/Processed/Clean/Participants/P7*/*/stationary_tasks/fixed_gaze/cam_face/*avi" \
            4 4 --output_file montage_cam_face_fixed_gaze_R1.mp4
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path_pattern",
        type=str,
        help="wildcard search pattern for videos to montage",
    )
    parser.add_argument("rows", type=int, help="number of rows")
    parser.add_argument("cols", type=int, help="number of columns")
    parser.add_argument("--fps", type=int, default=30, help="output video fps")
    parser.add_argument("--downsample", type=int, default=4, help="downsampling factor")
    parser.add_argument("--output_file", type=str, default="montage.mp4", help="output file name")
    args = parser.parse_args()

    video_list = glob.glob(args.video_path_pattern)
    random.shuffle(video_list)
    num_videos = args.rows * args.cols
    assert len(video_list) >= num_videos
    video_list = video_list[:num_videos]
    create_montage(video_list, (args.rows, args.cols), args.fps, args.downsample, args.output_file)
