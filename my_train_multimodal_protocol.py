"""
Multimodal training script supporting three input modes:
  flow  — optical flow only (2-ch),  reads flow_hdf5_frame_chunks
  video — RGB frames only   (3-ch),  reads video_hdf5_frame_chunks
  both  — flow + video fused (5-ch), reads both HDF5s

Model backbone: r3d18 or cnn  (same as protocol script)
For "both" mode the two streams are concatenated along the channel dim
before entering the backbone. The stem Conv3d in/out channels are
adjusted accordingly.

Usage:
    python my_train_multimodal_protocol.py --fold 0 --mode flow
    python my_train_multimodal_protocol.py --fold 0 --mode video
    python my_train_multimodal_protocol.py --fold 0 --mode both
"""

import csv
import random
import argparse
import json
import numpy as np
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
FLOW_HDF5_DIR  = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/flow_hdf5_frame_chunks")
VIDEO_HDF5_DIR = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/video_hdf5_frame_chunks")
FOLDS_CSV      = Path("/mnt/gs21/scratch/rubabfiz/repos/DBM/hail-datasets/hail_datasets/datasets/ddd_2024/folds.csv")

WANDB_PROJECT = None  # set to "DBM-OFlow" to enable

EXCLUDE_PARTICIPANTS = {"P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"}
SCENARIO_NAMES       = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]

# ── hyper-params ──────────────────────────────────────────────────────────────
MODEL           = "r3d18"   # "r3d18" or "cnn"
INPUT_MODE      = "video"    # "flow" | "video" | "both"  (overridden by --mode)

CHUNK_FRAMES    = 300
BUFFER_FRAMES   = 300
VAL_CHUNKS      = 4
CHUNK_STRIDE    = 300
TEMPORAL_STRIDE = 1

BATCH_SIZE  = 4
EPOCHS      = 10
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
NUM_WORKERS = 4

# ── channel counts per mode ───────────────────────────────────────────────────
IN_CHANNELS = {"flow": 2, "video": 3, "both": 5}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True


def make_save_root(mode, fold):
    tag = f"models_{MODEL}_{mode}_c{CHUNK_FRAMES}_b{BUFFER_FRAMES}_ts{TEMPORAL_STRIDE}_ep{EPOCHS}_lr{LR}_bs{BATCH_SIZE}"
    root = Path(f"./{tag}")
    root.mkdir(exist_ok=True)
    return root


def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


# ── data helpers ──────────────────────────────────────────────────────────────

def load_folds(folds_csv):
    participant_fold = {}
    with open(folds_csv) as f:
        for fold_idx, line in enumerate(csv.reader(f)):
            for pid in line:
                pid = pid.strip()
                if pid:
                    participant_fold[pid] = fold_idx
    return participant_fold


def scan_data(participant_fold, mode):
    """
    Scans whichever HDF5 dir(s) are relevant for `mode`.
    All three modes share the same keys (same scenarios were processed).
    Primary dir used for scanning: flow for "flow"/"both", video for "video".
    """
    primary_dir = VIDEO_HDF5_DIR if mode == "video" else FLOW_HDF5_DIR
    suffix      = "_video.h5"   if mode == "video" else "_flow.h5"
    dset_key    = "frames"      if mode == "video" else "flow"

    samples    = []
    min_frames = CHUNK_FRAMES + 2 * BUFFER_FRAMES

    for h5_path in sorted(primary_dir.glob(f"*{suffix}")):
        # stem is e.g. "P001_flow" or "P001_video"
        pid = h5_path.stem.rsplit("_", 1)[0]

        if pid in EXCLUDE_PARTICIPANTS or pid not in participant_fold:
            continue
        fold = participant_fold[pid]

        with h5py.File(h5_path, "r") as hf:
            for key in hf.keys():
                if "practice" in key:
                    continue
                n_frames = hf[key][dset_key].shape[0]
                label    = int(hf[key]["label"][()])
                if n_frames < min_frames:
                    continue
                n_train_chunks = max(1, (n_frames - 2 * BUFFER_FRAMES) // CHUNK_FRAMES)

                # build paths for both modalities (used in "both" mode)
                flow_h5  = FLOW_HDF5_DIR  / f"{pid}_flow.h5"
                video_h5 = VIDEO_HDF5_DIR / f"{pid}_video.h5"

                samples.append({
                    "participant":    pid,
                    "fold":           fold,
                    "label":          label,
                    "flow_h5":        flow_h5,
                    "video_h5":       video_h5,
                    "key":            key,
                    "n_frames":       n_frames,
                    "n_train_chunks": n_train_chunks,
                })

    return samples


def get_train_start(n_frames):
    low  = BUFFER_FRAMES
    high = n_frames - BUFFER_FRAMES - CHUNK_FRAMES
    return random.randint(low, max(low, high))


def get_val_starts(n_frames):
    starts = []
    for i in range(VAL_CHUNKS):
        s = n_frames - BUFFER_FRAMES - CHUNK_FRAMES - i * CHUNK_STRIDE
        if s >= BUFFER_FRAMES:
            starts.append(s)
    return starts


# ── dataset ───────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    def __init__(self, samples, train, mode):
        self.train = train
        self.mode  = mode
        self.items = []
        for s in samples:
            if train:
                for _ in range(s["n_train_chunks"]):
                    self.items.append((s["flow_h5"], s["video_h5"],
                                       s["key"], s["n_frames"], None, s["label"]))
            else:
                for start in get_val_starts(s["n_frames"]):
                    self.items.append((s["flow_h5"], s["video_h5"],
                                       s["key"], s["n_frames"], start, s["label"]))
        self._handles = {}

    def _get_handle(self, h5_path):
        k = str(h5_path)
        if k not in self._handles:
            self._handles[k] = h5py.File(h5_path, "r")
        return self._handles[k]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        flow_h5, video_h5, key, n_frames, start, label = self.items[idx]
        if self.train:
            start = get_train_start(n_frames)
        end = start + CHUNK_FRAMES

        tensors = []

        if self.mode in ("flow", "both"):
            hf   = self._get_handle(flow_h5)
            flow = hf[key]["flow"][start:end:TEMPORAL_STRIDE].astype(np.float32)
            flow = np.clip(flow, -20, 20) / 20.0          # (T, H, W, 2)
            flow = np.transpose(flow, (3, 0, 1, 2))        # (2, T, H, W)
            tensors.append(torch.from_numpy(flow))

        if self.mode in ("video", "both"):
            hf     = self._get_handle(video_h5)
            frames = hf[key]["frames"][start:end:TEMPORAL_STRIDE].astype(np.float32)
            frames = frames / 255.0                         # (T, H, W, 3)
            # ImageNet-style normalisation
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frames = (frames - mean) / std
            frames = np.transpose(frames, (3, 0, 1, 2))    # (3, T, H, W)
            tensors.append(torch.from_numpy(frames))

        x = torch.cat(tensors, dim=0)  # (2|3|5, T, H, W)
        return x, torch.tensor(label, dtype=torch.float32)


# ── models ────────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.BatchNorm3d(16),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32),
            nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1)).view(-1)


def build_r3d18(in_channels):
    from torchvision.models.video import r3d_18, R3D_18_Weights
    backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    old_conv = backbone.stem[0]
    new_conv = nn.Conv3d(in_channels, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                         padding=old_conv.padding, bias=old_conv.bias is not None)
    with torch.no_grad():
        # mean over RGB dim of pretrained weights → repeat to fill in_channels
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)         # (64,1,...)
        new_conv.weight.copy_(mean_w.expand(-1, in_channels, *mean_w.shape[2:]))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    backbone.stem[0] = new_conv
    backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(backbone.fc.in_features, 1))
    return backbone


def build_model(mode):
    ch = IN_CHANNELS[mode]
    if MODEL == "r3d18":
        return build_r3d18(ch)
    return SmallCNN(ch)


# ── training ──────────────────────────────────────────────────────────────────

def compute_metrics(gts, probs, threshold=0.5):
    gts   = np.array(gts)
    probs = np.array(probs)
    if len(gts) == 0:
        return {m: float("nan") for m in ("uar", "sensitivity", "specificity", "auc", "f1")}
    preds = (probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (gts == 1))
    tn = np.sum((preds == 0) & (gts == 0))
    fp = np.sum((preds == 1) & (gts == 0))
    fn = np.sum((preds == 0) & (gts == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    uar = ((sensitivity + specificity) / 2.0
           if not (np.isnan(sensitivity) or np.isnan(specificity)) else float("nan"))
    auc = roc_auc_score(gts, probs) if len(np.unique(gts)) > 1 else float("nan")
    f1  = f1_score(gts, preds, zero_division=0)
    return {"uar": uar, "sensitivity": sensitivity, "specificity": specificity,
            "auc": auc, "f1": f1}


def run_epoch(model, loader, criterion, optimizer=None, training=True, desc=""):
    model.train() if training else model.eval()
    total_loss, probs_all, gts_all = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=(DEVICE == "cuda")):
                logits = model(x).view(-1)
                loss   = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
            gts_all.extend(y.cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = max(len(loader), 1)
    return total_loss / n, compute_metrics(gts_all, probs_all)


def make_loader(samples, train, mode):
    ds = MultimodalDataset(samples, train=train, mode=mode)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                      prefetch_factor=2 if NUM_WORKERS > 0 else None,
                      persistent_workers=(NUM_WORKERS > 0))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Single test fold 0-4. Omit to run all 5.")
    parser.add_argument("--mode", choices=["flow", "video", "both"],
                        default=INPUT_MODE,
                        help="Input modality: flow, video, or both (early fusion).")
    args = parser.parse_args()

    mode         = args.mode
    folds_to_run = list(range(5)) if args.fold is None else [args.fold]
    save_root    = make_save_root(mode, args.fold)
    log_file     = save_root / "training_log.txt"

    def L(msg): log(msg, log_file)

    L(f"\n{'='*60}")
    L(f"MODE={mode}  MODEL={MODEL}  DEVICE={DEVICE}  "
      f"CHUNK={CHUNK_FRAMES}fr  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  folds={folds_to_run}")
    L(f"{'='*60}")

    participant_fold = load_folds(FOLDS_CSV)
    all_samples      = scan_data(participant_fold, mode)

    L(f"\nTotal sequences: {len(all_samples)}")
    L(f"  Intoxicated: {sum(s['label']==1 for s in all_samples)}")
    L(f"  Sober:       {sum(s['label']==0 for s in all_samples)}")

    fold_test_uars = []

    for test_fold in folds_to_run:
        val_fold = (test_fold + 1) % 5
        train_s  = [s for s in all_samples if s["fold"] != test_fold and s["fold"] != val_fold]
        val_s    = [s for s in all_samples if s["fold"] == val_fold]
        test_s   = [s for s in all_samples if s["fold"] == test_fold]

        intox_s  = [s for s in train_s if s["label"] == 1]
        sober_s  = [s for s in train_s if s["label"] == 0]
        sober_s  = random.sample(sober_s, min(len(intox_s), len(sober_s)))
        train_s  = intox_s + sober_s
        random.shuffle(train_s)

        L(f"\n{'='*60}")
        L(f"FOLD {test_fold+1}/5  (test={test_fold}, val={val_fold})")
        L(f"  Train: {len(train_s)} ({len(intox_s)} intox + {len(sober_s)} sober)  "
          f"Val: {len(val_s)}  Test: {len(test_s)}")

        train_loader = make_loader(train_s, train=True,  mode=mode)
        val_loader   = make_loader(val_s,   train=False, mode=mode)
        test_loader  = make_loader(test_s,  train=False, mode=mode)

        model     = build_model(mode).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        fold_dir     = save_root / f"fold_{test_fold+1}"
        fold_dir.mkdir(exist_ok=True)
        best_val_uar = -1.0
        best_epoch   = 0

        for epoch in range(EPOCHS):
            tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer,
                                      training=True,  desc=f"F{test_fold+1} Ep{epoch+1} train")
            vl_loss, vl_m = run_epoch(model, val_loader,   criterion,
                                      training=False, desc=f"F{test_fold+1} Ep{epoch+1} val")
            scheduler.step()

            L(f"  Ep{epoch+1:02d} | train loss={tr_loss:.4f} uar={tr_m['uar']:.4f} | "
              f"val loss={vl_loss:.4f} uar={vl_m['uar']:.4f} "
              f"sens={vl_m['sensitivity']:.4f} spec={vl_m['specificity']:.4f} "
              f"auc={vl_m['auc']:.4f}")

            torch.save(model.state_dict(), fold_dir / f"epoch_{epoch+1:02d}.pth")
            uar_val = vl_m["uar"] if not np.isnan(vl_m["uar"]) else -1.0
            if uar_val > best_val_uar:
                best_val_uar = uar_val
                best_epoch   = epoch + 1
                torch.save(model.state_dict(), fold_dir / "best_model.pth")
                L(f"    *** new best val UAR={best_val_uar:.4f} (epoch {best_epoch}) ***")

        model.load_state_dict(torch.load(fold_dir / "best_model.pth"))
        _, test_m = run_epoch(model, test_loader, criterion,
                              training=False, desc=f"F{test_fold+1} test")

        L(f"\n  FOLD {test_fold+1} TEST (best epoch={best_epoch}):")
        L(f"    UAR={test_m['uar']:.4f}  sens={test_m['sensitivity']:.4f}  "
          f"spec={test_m['specificity']:.4f}  AUC={test_m['auc']:.4f}  F1={test_m['f1']:.4f}")

        fold_test_uars.append(test_m["uar"])
        result = {"fold": test_fold, "best_epoch": best_epoch, "mode": mode, **test_m}
        with open(fold_dir / "test_results.json", "w") as fp:
            json.dump(result, fp, indent=2)

    if len(folds_to_run) > 1:
        L(f"\n{'='*60}")
        L("FINAL 5-FOLD TEST RESULTS")
        for fi, uar in zip(folds_to_run, fold_test_uars):
            L(f"  Fold {fi+1}: UAR={uar:.4f}")
        L(f"  Mean UAR: {np.mean(fold_test_uars):.4f}  Std: {np.std(fold_test_uars):.4f}")


if __name__ == "__main__":
    main()
