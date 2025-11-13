import argparse
import os

import cv2
import pandas as pd


def generate_gaze_video(
    processed_dir,
    output_fps=None,
    output_file=None,
    render_last_n_seconds=1000,
    cam_name="cam_front",
):
    """render_last_n_seconds allows to capture only last N seconds of a scenario video"""

    # assumes output path structure
    frame_timestamp_csv = os.path.join(processed_dir, f"{cam_name}/frame_timing.csv")
    eye_tracker_csv = os.path.join(processed_dir, "sim_bag/experiment_tobii_frame.csv")
    video_file = os.path.join(processed_dir, f"{cam_name}/video.avi")
    if not output_file:
        output_file = os.path.join(processed_dir, f"{cam_name}/video_gaze.avi")
    print("Generating gaze video...")
    print(f"  Input: {video_file}")
    print(f"  Output: {output_file}")

    # load timestamp data and remove NaNs from tobii
    frame_timestamps = pd.read_csv(frame_timestamp_csv)
    eye_tracker = pd.read_csv(eye_tracker_csv)
    eye_tracker = eye_tracker.dropna(
        subset=[
            "tobii_left_eye_gaze_pt_in_display_x",
            "tobii_left_eye_gaze_pt_in_display_y",
            "tobii_right_eye_gaze_pt_in_display_x",
            "tobii_right_eye_gaze_pt_in_display_y",
        ]
    )

    cap = cv2.VideoCapture(video_file)
    if output_fps:
        fps = output_fps
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)

    # find field name ending in " log time"
    time_fields = frame_timestamps.columns[frame_timestamps.columns.str.endswith(" log time")]
    assert len(time_fields) == 1
    log_time = time_fields[0]

    # find minimum starting time stamp
    end_time = frame_timestamps.iloc[-1][log_time]
    start_time = end_time - render_last_n_seconds

    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    circle_radius = int(max(3, frame_height / 40))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if frame_index == len(frame_timestamps) - 1 or not ret:
            break

        # find eye tracker data within current frame's log_time_ns
        log_time_current = frame_timestamps.iloc[frame_index][log_time]
        log_time_next = frame_timestamps.iloc[frame_index + 1][log_time]

        if log_time_current < start_time:
            frame_index += 1
            continue

        eye_tracker_rows = eye_tracker[
            (eye_tracker["tobii log time"] >= log_time_current) & (eye_tracker["tobii log time"] < log_time_next)
        ]

        # convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # create overlay for output
        overlay = gray_frame.copy()

        # draw left and right gaze points as transparent green circles
        for _, row in eye_tracker_rows.iterrows():
            x = int(row["tobii_left_eye_gaze_pt_in_display_x"] * frame_width)
            y = int(row["tobii_left_eye_gaze_pt_in_display_y"] * frame_height)
            if cam_name == "cam_face":
                x = frame_width - x  # invert x axis for viz
            cv2.circle(gray_frame, (x, y), circle_radius, (0, 255, 0, 128), -1)
            x = int(row["tobii_right_eye_gaze_pt_in_display_x"] * frame_width)
            y = int(row["tobii_right_eye_gaze_pt_in_display_y"] * frame_height)
            if cam_name == "cam_face":
                x = frame_width - x  # invert x axis for viz
            cv2.circle(gray_frame, (x, y), circle_radius, (0, 255, 0, 128), -1)

        # blend and write
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, gray_frame, 1 - alpha, 0, gray_frame)
        out.write(gray_frame)

        frame_index += 1

    cap.release()
    out.release()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="/data/motion-simulator-logs/Processed/Clean/Participants/P720/R2/driving/8a_pedestrian_pop_out/statement_task",
        help="path to the processed scenario (assumes the Processed/Clean/Participants file structure)",
    )
    parser.add_argument("--output_fps", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None, help="non-standard output path for video")
    parser.add_argument("--render_last_n_seconds", type=int, default=1000, help="render last N seconds of video")
    parser.add_argument("--cam_name", type=str, default="cam_front", help="render over a different video")
    args = parser.parse_args()

    generate_gaze_video(
        args.processed_dir,
        args.output_fps,
        args.output_file,
        args.render_last_n_seconds,
        args.cam_name,
    )
