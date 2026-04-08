import os
import cv2
from pathlib import Path

# ==========================
# Paths
# ==========================
PARENT_DIR = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/Full/Resampled_previous_10/Participants")
OUTPUT_DIR = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/frames_resized_224h")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HEIGHT = 224
JPEG_QUALITY = 85


def resize_keep_aspect(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized


def extract_frames_from_video(video_path, output_folder):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every other frame (reduce to 5 FPS)
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        frame = resize_keep_aspect(frame, TARGET_HEIGHT)

        frame_name = f"{saved_idx:05d}.jpg"
        cv2.imwrite(
            str(output_folder / frame_name),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"✅ {video_path.name} → {saved_idx} frames saved")


# ==========================
# Walk through all .avi files
# ==========================
for video_path in PARENT_DIR.rglob("*.avi"):

    parts = video_path.parts

    try:
        participant_id = parts[parts.index("Participants") + 1]
        r_folder = parts[parts.index(participant_id) + 1]

        driving_index = parts.index("driving")
        task_folder = parts[driving_index + 1]
        subtask_folder = parts[driving_index + 2]

    except Exception:
        print(f"⚠️ Skipping malformed path: {video_path}")
        continue

    combined_name = f"{r_folder}+{task_folder}+{subtask_folder}"

    save_dir = OUTPUT_DIR / participant_id / combined_name
    save_dir.mkdir(parents=True, exist_ok=True)

    extract_frames_from_video(video_path, save_dir)

print("🎉 Done extracting all videos.")