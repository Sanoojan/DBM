import csv
import numpy as np
import cv2
import h5py
from pathlib import Path
from tqdm import tqdm

DATA_ROOT      = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/No-Video/Resampled_previous_10/Participants")
FLOW_HDF5_DIR  = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/flow_hdf5_112")
VIDEO_HDF5_DIR = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/video_hdf5_112")
ANOMALY_CSV    = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/No-Video/idd_annotation.csv")
FLOW_HDF5_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_HDF5_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE        = {"P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"}
TARGET_H       = 112
TARGET_W       = 224
SCENARIO_NAMES = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]


def load_anomalies(anomaly_csv):
    anomaly_map = {}
    with open(anomaly_csv) as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            if not row:
                continue
            pid = row[0].strip()
            if not pid or pid[0] not in ("P", "7"):
                continue
            bad = set()
            for col_idx, sname in enumerate(SCENARIO_NAMES):
                cell = row[1 + col_idx].strip() if 1 + col_idx < len(row) else ""
                if any(len([x for x in p.strip().split("-") if x]) == 2 for p in cell.split(",")):
                    bad.add(sname)
            anomaly_map[pid] = bad
    return anomaly_map


def scenario_has_anomaly(pid, scenario_dir_name, anomaly_map):
    if pid not in anomaly_map:
        return False
    return scenario_dir_name.split("-")[0] in anomaly_map[pid]


def resize_crop(frame, target_h, target_w):
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def process_participant(p_dir, anomaly_map):
    pid       = p_dir.name
    flow_out  = FLOW_HDF5_DIR  / f"{pid}_flow.h5"
    video_out = VIDEO_HDF5_DIR / f"{pid}_video.h5"

    if flow_out.exists() and video_out.exists():
        print(f"  skip {pid} (already done)")
        return

    flow_tmp  = FLOW_HDF5_DIR  / f"{pid}_flow.h5.tmp"
    video_tmp = VIDEO_HDF5_DIR / f"{pid}_video.h5.tmp"
    is_p      = pid.startswith("P")

    with h5py.File(flow_tmp, "w") as hf_flow, h5py.File(video_tmp, "w") as hf_video:
        for rnd in ["R1", "R2"]:
            label   = 1 if (is_p and rnd == "R2") else 0
            driving = p_dir / rnd / "driving"
            if not driving.exists():
                continue

            for scenario_dir in sorted(driving.iterdir()):
                if not scenario_dir.is_dir():
                    continue
                scenario = scenario_dir.name
                if "practice" in scenario:
                    continue
                if scenario_has_anomaly(pid, scenario, anomaly_map):
                    print(f"    skip anomaly {pid}/{rnd}/{scenario}")
                    continue

                for task_dir in sorted(scenario_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    task        = task_dir.name
                    video_path  = task_dir / "cam_front" / "video.avi"
                    timing_path = task_dir / "cam_front" / "frame_timing.csv"
                    if not video_path.exists() or not timing_path.exists():
                        continue

                    frame_indices = []
                    with open(timing_path) as f:
                        for row in csv.DictReader(f):
                            frame_indices.append(int(float(row["resampled_frame"])))
                    if len(frame_indices) < 2:
                        continue

                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"    can't open {video_path}")
                        continue

                    target_set = set(frame_indices)
                    bgr_map    = {}
                    gray_map   = {}
                    avi_idx    = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if avi_idx in target_set:
                            frame             = resize_crop(frame, TARGET_H, TARGET_W)
                            bgr_map[avi_idx]  = frame
                            gray_map[avi_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        avi_idx += 1
                    cap.release()

                    valid = [fi for fi in frame_indices if fi in bgr_map]
                    if len(valid) < 2:
                        continue

                    grays = [gray_map[fi] for fi in valid]
                    flows = []
                    for i in range(1, len(grays)):
                        flows.append(cv2.calcOpticalFlowFarneback(
                            grays[i-1], grays[i], None,
                            pyr_scale=0.5, levels=3, winsize=15,
                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                        ))

                    flow_arr  = np.stack(flows).astype(np.float32)
                    video_arr = np.stack([bgr_map[fi] for fi in valid])

                    key = f"{rnd}+{scenario}+{task}"

                    grp = hf_flow.create_group(key)
                    grp.create_dataset("flow",  data=flow_arr,  compression="lzf",
                                       chunks=(flow_arr.shape[0],  *flow_arr.shape[1:]))
                    grp.create_dataset("label", data=np.int8(label))

                    grp_v = hf_video.create_group(key)
                    grp_v.create_dataset("video", data=video_arr, compression="lzf",
                                         chunks=(video_arr.shape[0], *video_arr.shape[1:]))
                    grp_v.create_dataset("label", data=np.int8(label))

                    print(f"    {pid}/{key}: {flow_arr.shape[0]} flow, {video_arr.shape[0]} video frames")

    flow_tmp.rename(flow_out)
    video_tmp.rename(video_out)
    print(f"  ✓ {pid} done")


def main():
    anomaly_map  = load_anomalies(ANOMALY_CSV)
    participants = sorted(p for p in DATA_ROOT.iterdir()
                          if p.is_dir() and p.name not in EXCLUDE)
    print(f"Processing {len(participants)} participants")

    for p_dir in tqdm(participants, desc="Participants"):
        process_participant(p_dir, anomaly_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
