import cv2
import numpy as np

from utils.video import colorize_depth_frame, colorize_segmentation_frame


def get_video_with_gaze_frame(sample, video_name, current_resample_ns, args, within_time_s=0.1):
    # Get video and timing
    video = sample.features[video_name]

    # Get the frame
    video.init_frame_reader()
    frame_data = video.frame_reader.read_at_time(current_resample_ns)
    if frame_data is None:
        image_data = np.zeros(video["image_size"])
    else:
        image_data = frame_data.image_data

    # Process depth frame data
    if video_name == "cam_depth":
        # Compute log depth
        image_data /= 1000.0
        logdepth = np.ones(image_data.shape) + (np.log(image_data) / 5.70378)
        logdepth = np.clip(logdepth, 0.0, 1.0)

        # Colorize video
        image_data = colorize_depth_frame(logdepth)

    # Process semantic segmentation frame data
    if video_name == "cam_seg":
        image_data = colorize_segmentation_frame(image_data[:, :, 0])

    if not args.no_gaze:
        # Get the gaze position on the screen
        lower_time_limit_ns = current_resample_ns - int(within_time_s * 1e9)
        tabular_subset = sample.features["tobii"].table.loc[lower_time_limit_ns:current_resample_ns]
        tobii_names = ["tobii_left_eye_gaze_pt_in_display_" + axis for axis in ["x", "y"]]
        if tabular_subset.shape[0] == 0:
            return image_data
        gaze_in_screen = tabular_subset[tobii_names].iloc[-1].to_numpy()
        if np.any(np.isnan(gaze_in_screen)):
            return image_data
        gaze_in_screen[0] *= image_data.shape[1]
        gaze_in_screen[1] *= image_data.shape[0]

        # Add the gaze
        gaze_in_screen = np.round(gaze_in_screen)
        gaze_pixel = (int(gaze_in_screen[0]), int(gaze_in_screen[1]))
        plot_yellow = np.array([0.0, 0.8, 1.0]) * 255.0
        overlay = image_data.copy()
        overlay = cv2.circle(overlay, gaze_pixel, 10, plot_yellow, 5)
        overlay = cv2.circle(overlay, gaze_pixel, 12, np.sqrt(plot_yellow), 2)
        alpha = 0.75
        image_data = cv2.addWeighted(overlay, alpha, image_data, 1 - alpha, 0)

    return image_data
