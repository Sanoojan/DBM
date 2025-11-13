import os

import cv2
import numpy as np
import pandas as pd


def shrink_video(sample, output_writer):
    # Check that video exists
    if "cam_front" not in sample.features:
        return

    # Set up the video output
    input_video = sample.features["cam_front"]
    frame_reader = input_video.get_frame_reader()
    fourcc = int(frame_reader.cap.get(cv2.CAP_PROP_FOURCC))
    fps = input_video.fps
    image_size = frame_reader.image_size

    frame_writer = output_writer.get_video_frame_writer(
        "cam_front_scale4", ".avi", fourcc, fps, (image_size[1], image_size[0])
    )

    # Loop through video frames one at a time
    for frame in frame_reader.read_frames():
        # Image data stored in frame.image_data - write back out since already scaled
        frame_writer.write(frame.time, frame.image_data, frame.timing_row)

    # Close and write the video
    frame_writer.close()
    frame_reader.close()
