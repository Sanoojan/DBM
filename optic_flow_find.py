import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================
# Paths
# ==========================
FRAME_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/frames_resized_224h")
FLOW_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/optical_flow_224h")

FLOW_ROOT.mkdir(parents=True, exist_ok=True)


def compute_and_save_flow(frame_folder, flow_folder):
    flow_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frame_folder.glob("*.jpg"))
    # breakpoint()

    if len(frame_paths) < 2:
        return

    prev_frame = cv2.imread(str(frame_paths[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frame_paths)):
        curr_frame = cv2.imread(str(frame_paths[i]))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Save as float32 numpy array
        np.save(flow_folder / f"{i-1:05d}.npy", flow.astype(np.float32))

        prev_gray = curr_gray

    print(f"✅ Done: {frame_folder}")


# ==========================
# Traverse all video folders
# ==========================
for participant_dir in FRAME_ROOT.iterdir():
    if not participant_dir.is_dir():
        continue

    for video_dir in participant_dir.iterdir():
        if not video_dir.is_dir():
            continue
        

        relative_path = video_dir.relative_to(FRAME_ROOT)
        flow_output_dir = FLOW_ROOT / relative_path
        
        if flow_output_dir.exists():
            print(f"⚠️  Skipping (already exists): {flow_output_dir}")
            continue

        compute_and_save_flow(video_dir, flow_output_dir)

print("🎉 All optical flow computed.")