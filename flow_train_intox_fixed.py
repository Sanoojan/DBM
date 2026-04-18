import csv
import random
import numpy as np
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

FLOW_HDF5_DIR  = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/flow_hdf5_112")
VIDEO_HDF5_DIR = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/video_hdf5_112")
FOLDS_CSV      = Path("/mnt/gs21/scratch/rubabfiz/repos/DBM/hail-datasets/hail_datasets/datasets/ddd_2024/folds.csv")
ANOMALY_CSV    = Path("/mnt/scratch/rubabfiz/repos/DBM/dataset/Vehicle/No-Video/idd_annotation.csv")
SAVE_ROOT      = Path("./flow_intox_models2")
SAVE_ROOT.mkdir(exist_ok=True)

USE_VIDEO    = False  # False = flow only (2ch), "video_only" (3ch), "fused" (5ch)
WANDB_PROJECT = "DBM-OFlow"   # set to None to disable wandb

EXCLUDE_PARTICIPANTS = {"P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"}
SCENARIO_NAMES       = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]

CHUNK_FRAMES  = 300
BUFFER_FRAMES = 300
VAL_CHUNKS    = 4
CHUNK_STRIDE  = 300

TEMPORAL_STRIDE = 1    
BATCH_SIZE      = 2
EPOCHS          = 7
LR              = 1e-3
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42
NUM_WORKERS     = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

LOG_FILE = SAVE_ROOT / "training_log.txt"


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


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


def has_anomaly(pid, hdf5_key, anomaly_map):
    if pid not in anomaly_map:
        return False
    short_name = hdf5_key.split("+")[1].split("-")[0]
    return short_name in anomaly_map[pid]


def load_folds(folds_csv):
    participant_fold = {}
    with open(folds_csv) as f:
        for fold_idx, line in enumerate(csv.reader(f)):
            for pid in line:
                pid = pid.strip()
                if pid:
                    participant_fold[pid] = fold_idx
    return participant_fold


def scan_data(flow_hdf5_dir, participant_fold, anomaly_map, video_hdf5_dir=None):
    import time
    samples    = []
    min_frames = CHUNK_FRAMES + 2 * BUFFER_FRAMES
    t_scan_start = time.perf_counter()

    all_paths = sorted(flow_hdf5_dir.glob("*_flow.h5"))
    for flow_path in all_paths:
        pid = flow_path.stem[:-5]
        if pid in EXCLUDE_PARTICIPANTS:
            continue
        if pid not in participant_fold:
            log(f"  WARNING: {pid} not in folds.csv — skipping")
            continue
        fold = participant_fold[pid]

        video_path = None
        if video_hdf5_dir is not None:
            candidate = video_hdf5_dir / f"{pid}_video.h5"
            if candidate.exists():
                video_path = candidate
            else:
                log(f"  WARNING: {pid}_video.h5 not found — skipping")
                continue

        t0 = time.perf_counter()
        with h5py.File(flow_path, "r") as hf:
            for key in hf.keys():
                if "practice" in key:
                    continue
                if has_anomaly(pid, key, anomaly_map):
                    continue
                n_frames = hf[key]["flow"].shape[0]
                label    = int(hf[key]["label"][()])
                if n_frames < min_frames:
                    continue
                n_train_chunks = max(1, (n_frames - 2 * BUFFER_FRAMES) // CHUNK_FRAMES)
                samples.append({
                    "participant":   pid,
                    "fold":          fold,
                    "label":         label,
                    "h5_path":       flow_path,
                    "video_h5_path": video_path,
                    "key":           key,
                    "n_frames":      n_frames,
                    "n_train_chunks": n_train_chunks,
                })
        print(f"  scan {pid}: {time.perf_counter()-t0:.2f}s  ({len([s for s in samples if s['participant']==pid])} scenarios)")

    print(f"scan_data total: {time.perf_counter()-t_scan_start:.1f}s  →  {len(samples)} samples")
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


class FlowDataset(Dataset):
    def __init__(self, samples, train):
        self.train = train
        self.items = []
        for s in samples:
            if train:
                for _ in range(s["n_train_chunks"]):
                    self.items.append((s["h5_path"], s["video_h5_path"],
                                       s["key"], s["n_frames"], None, s["label"]))
            else:
                for start in get_val_starts(s["n_frames"]):
                    self.items.append((s["h5_path"], s["video_h5_path"],
                                       s["key"], s["n_frames"], start, s["label"]))
        self._handles = {}

    def _get_handle(self, h5_path):
        path_str = str(h5_path)
        if path_str not in self._handles:
            self._handles[path_str] = h5py.File(h5_path, "r")
        return self._handles[path_str]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        flow_path, video_path, key, n_frames, start, label = self.items[idx]
        if self.train:
            start = get_train_start(n_frames)
        hf_flow = self._get_handle(flow_path)

        if USE_VIDEO == "video_only":
            hf_video = self._get_handle(video_path)
            video = hf_video[key]["video"][start + 1 : start + 1 + CHUNK_FRAMES : TEMPORAL_STRIDE]
            video = np.transpose(video, (3, 0, 1, 2)).astype(np.float32) / 255.0
            x = torch.from_numpy(video)
        elif USE_VIDEO == "fused":
            flow  = hf_flow[key]["flow"][start : start + CHUNK_FRAMES : TEMPORAL_STRIDE]
            flow  = np.transpose(flow, (3, 0, 1, 2)).astype(np.float32)
            flow  = np.clip(flow, -20, 20) / 20.0
            hf_video = self._get_handle(video_path)
            video = hf_video[key]["video"][start + 1 : start + 1 + CHUNK_FRAMES : TEMPORAL_STRIDE]
            video = np.transpose(video, (3, 0, 1, 2)).astype(np.float32) / 255.0
            x = torch.from_numpy(np.concatenate([flow, video], axis=0))
        else:
            flow = hf_flow[key]["flow"][start : start + CHUNK_FRAMES : TEMPORAL_STRIDE]
            flow = np.transpose(flow, (3, 0, 1, 2)).astype(np.float32)
            flow = np.clip(flow, -20, 20) / 20.0
            x    = torch.from_numpy(flow)

        return x, torch.tensor(label, dtype=torch.float32)


def _input_channels():
    if USE_VIDEO == "video_only":
        return 3
    elif USE_VIDEO == "fused":
        return 5
    return 2


def build_model(freeze_backbone=False):
    from torchvision.models.video import r3d_18, R3D_18_Weights
    backbone    = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    in_channels = _input_channels()

    if in_channels != 3:
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                             padding=old_conv.padding, bias=old_conv.bias is not None)
        with torch.no_grad():
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_w.expand_as(new_conv.weight))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        backbone.stem[0] = new_conv

    backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(backbone.fc.in_features, 1))

    if freeze_backbone:
        for name, param in backbone.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

    return backbone


def compute_metrics(gts, probs, threshold=0.5):
    gts   = np.array(gts)
    probs = np.array(probs)
    if len(gts) == 0:
        return {"uar": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "auc": float("nan"), "f1": float("nan")}
    preds = (probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (gts == 1))
    tn = np.sum((preds == 0) & (gts == 0))
    fp = np.sum((preds == 1) & (gts == 0))
    fn = np.sum((preds == 0) & (gts == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    uar = (sensitivity + specificity) / 2.0 if not (np.isnan(sensitivity) or np.isnan(specificity)) else float("nan")
    auc = roc_auc_score(gts, probs) if len(np.unique(gts)) > 1 else float("nan")
    f1  = f1_score(gts, preds, zero_division=0)
    return {"uar": uar, "sensitivity": sensitivity, "specificity": specificity,
            "auc": auc, "f1": f1}


def run_epoch(model, loader, criterion, optimizer=None, training=True):
    import time
    from tqdm import tqdm
    model.train() if training else model.eval()
    total_loss, probs_all, gts_all = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()

    t_load_total = t_forward_total = t_backward_total = 0.0
    t_batch_start = time.perf_counter()

    desc = "train" if training else "val/test"
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for x, y in pbar:
            t_load = time.perf_counter() - t_batch_start
            t_load_total += t_load

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            t_fwd_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=(DEVICE == "cuda")):
                logits = model(x).view(-1)
                loss   = criterion(logits, y)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.perf_counter() - t_fwd_start
            t_forward_total += t_fwd

            t_bwd_start = time.perf_counter()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
            t_bwd = time.perf_counter() - t_bwd_start
            t_backward_total += t_bwd

            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
            gts_all.extend(y.cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             load=f"{t_load:.1f}s", fwd=f"{t_fwd:.2f}s", bwd=f"{t_bwd:.2f}s")
            t_batch_start = time.perf_counter()

    n = max(len(loader), 1)
    log(f"  [epoch totals] load={t_load_total:.1f}s  fwd={t_forward_total:.1f}s  bwd={t_backward_total:.1f}s  "
        f"avg/batch: load={t_load_total/n:.2f}s  fwd={t_forward_total/n:.2f}s  bwd={t_backward_total/n:.2f}s")

    return total_loss / n, compute_metrics(gts_all, probs_all)


def make_loader(samples, train):
    ds = FlowDataset(samples, train=train)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                      prefetch_factor=2 if NUM_WORKERS > 0 else None,
                      persistent_workers=(NUM_WORKERS > 0))


def main():
    import wandb as wb
    run_cfg = dict(chunk_frames=CHUNK_FRAMES, buffer_frames=BUFFER_FRAMES,
                   temporal_stride=TEMPORAL_STRIDE, batch_size=BATCH_SIZE,
                   epochs=EPOCHS, lr=LR, use_video=USE_VIDEO, seed=SEED)
    if WANDB_PROJECT:
        wb.init(project=WANDB_PROJECT, config=run_cfg)

    log(f"\n{'='*60}")
    log(f"DEVICE={DEVICE}  CHUNK={CHUNK_FRAMES}fr  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  USE_VIDEO={USE_VIDEO}")
    log(f"{'='*60}")

    participant_fold = load_folds(FOLDS_CSV)
    print("Folds loaded:", {fold: sum(1 for p in participant_fold if participant_fold[p] == fold) for fold in range(5)})
    anomaly_map      = load_anomalies(ANOMALY_CSV)
    print("Anomalies loaded:", {pid: len(anomaly_map[pid]) for pid in anomaly_map})
    video_dir        = VIDEO_HDF5_DIR if USE_VIDEO else None
    all_samples      = scan_data(FLOW_HDF5_DIR, participant_fold, anomaly_map, video_hdf5_dir=video_dir)
    log(f"\nTotal scenarios: {len(all_samples)}")
    log(f"  Intoxicated: {sum(s['label']==1 for s in all_samples)}")
    log(f"  Sober: {sum(s['label']==0 for s in all_samples)}")
    log(f"  Participants: {sorted(set(s['participant'] for s in all_samples))}")

    fold_test_uars = []

    for test_fold in range(1):
        val_fold = (test_fold + 1) % 5
        train_s  = [s for s in all_samples if s["fold"] != test_fold and s["fold"] != val_fold]
        val_s    = [s for s in all_samples if s["fold"] == val_fold]
        test_s   = [s for s in all_samples if s["fold"] == test_fold]

        log(f"\n{'='*60}")
        log(f"FOLD {test_fold+1}/5  (test={test_fold}, val={val_fold})")
        log(f"  Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")
        pos = sum(s["label"] for s in train_s)
        neg = len(train_s) - pos
        log(f"  Train pos(intox):{pos}  neg(sober):{neg} (before balancing)")

        intox_s  = [s for s in train_s if s["label"] == 1]
        sober_s  = [s for s in train_s if s["label"] == 0]
        sober_s  = random.sample(sober_s, min(len(intox_s), len(sober_s)))
        train_s  = intox_s + sober_s
        random.shuffle(train_s)
        log(f"  Train after balancing: {len(intox_s)} intox + {len(sober_s)} sober = {len(train_s)}")

        criterion    = nn.BCEWithLogitsLoss()
        train_loader = make_loader(train_s, train=True)
        val_loader   = make_loader(val_s,   train=False)
        test_loader  = make_loader(test_s,  train=False)

        model     = build_model(freeze_backbone=False).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        fold_dir     = SAVE_ROOT / f"fold_{test_fold+1}"
        fold_dir.mkdir(exist_ok=True)
        best_val_uar = -1.0
        best_epoch   = 0

        for epoch in range(EPOCHS):
            print(f"  Epoch {epoch+1}/{EPOCHS}")
            tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, training=True)
            vl_loss, vl_m = run_epoch(model, val_loader,   criterion, training=False)
            scheduler.step()

            log(f"  Ep{epoch+1:02d} | train loss={tr_loss:.4f} uar={tr_m['uar']:.4f} auc={tr_m['auc']:.4f} | "
                f"val loss={vl_loss:.4f} uar={vl_m['uar']:.4f} "
                f"sens={vl_m['sensitivity']:.4f} spec={vl_m['specificity']:.4f} auc={vl_m['auc']:.4f}")

            if WANDB_PROJECT:
                wb.log({
                    "fold": test_fold + 1, "epoch": epoch + 1,
                    "train/loss": tr_loss, "train/uar": tr_m["uar"],
                    "train/auc": tr_m["auc"], "train/f1": tr_m["f1"],
                    "val/loss": vl_loss, "val/uar": vl_m["uar"],
                    "val/sensitivity": vl_m["sensitivity"],
                    "val/specificity": vl_m["specificity"],
                    "val/auc": vl_m["auc"], "val/f1": vl_m["f1"],
                })

            torch.save(model.state_dict(), fold_dir / f"epoch_{epoch+1:02d}.pth")
            if vl_m["uar"] > best_val_uar:
                best_val_uar = vl_m["uar"]
                best_epoch   = epoch + 1
                torch.save(model.state_dict(), fold_dir / "best_model.pth")
                log(f"    *** new best val UAR={best_val_uar:.4f} (epoch {best_epoch}) ***")

        model.load_state_dict(torch.load(fold_dir / "best_model.pth"))
        _, test_m = run_epoch(model, test_loader, criterion, training=False)

        log(f"\n  FOLD {test_fold+1} TEST (best epoch={best_epoch}):")
        log(f"    UAR={test_m['uar']:.4f}  sens={test_m['sensitivity']:.4f}  "
            f"spec={test_m['specificity']:.4f}  AUC={test_m['auc']:.4f}  F1={test_m['f1']:.4f}")
        if WANDB_PROJECT:
            wb.log({f"test/fold{test_fold+1}_uar": test_m["uar"],
                    f"test/fold{test_fold+1}_auc": test_m["auc"],
                    f"test/fold{test_fold+1}_sensitivity": test_m["sensitivity"],
                    f"test/fold{test_fold+1}_specificity": test_m["specificity"],
                    f"test/fold{test_fold+1}_f1": test_m["f1"]})
        fold_test_uars.append(test_m["uar"])

    log(f"\n{'='*60}")
    log("FINAL 5-FOLD TEST RESULTS")
    for i, uar in enumerate(fold_test_uars, 1):
        log(f"  Fold {i}: UAR={uar:.4f}")
    log(f"  Mean UAR: {np.mean(fold_test_uars):.4f}  Std: {np.std(fold_test_uars):.4f}")
    if WANDB_PROJECT:
        wb.log({"test/mean_uar": float(np.mean(fold_test_uars)),
                "test/std_uar":  float(np.std(fold_test_uars))})
        wb.finish()


if __name__ == "__main__":
    main()
