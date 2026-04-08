import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# =========================================================
# CONFIG
# =========================================================
FLOW_ROOT   = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/optical_flow_224h")
SAVE_ROOT   = Path("./flow_only_models_opt")
SAVE_ROOT.mkdir(exist_ok=True)

T           = 200
INF_T       = 1000
BATCH_SIZE  = 8        # ↑ from 4 — better GPU utilization
EPOCHS      = 20
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
NUM_WORKERS = 4        # ↑ from 0 — parallel data loading
PREFETCH    = 2        # prefetch_factor for DataLoader

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enable cuDNN auto-tuner for fixed input sizes
torch.backends.cudnn.benchmark = True

# =========================================================
# LOGGING HELPER
# =========================================================
LOG_FILE = SAVE_ROOT / "training_log.txt"

def log(msg, also_print=True):
    if also_print:
        print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# =========================================================
# ORGANIZE DATA
# =========================================================
def organize_data():
    participant_videos = defaultdict(list)
    participant_labels = {}

    all_dirs = sorted(FLOW_ROOT.iterdir())
    log(f"\nScanning {len(all_dirs)} participant directories...")

    for participant_dir in tqdm(all_dirs, desc="Scanning participants"):
        if not participant_dir.is_dir():
            continue

        participant_id = participant_dir.name
        intoxicated_flag = 0

        for flow_video_dir in sorted(participant_dir.iterdir()):
            if not flow_video_dir.is_dir():
                continue

            video_name = flow_video_dir.name

            if participant_id.startswith("P") and video_name.startswith("R2"):
                label = 1
                intoxicated_flag = 1
            else:
                label = 0

            flow_files = sorted(flow_video_dir.glob("*.npy"))
            if len(flow_files) < T:
                continue

            participant_videos[participant_id].append((flow_video_dir, video_name, label))

        if participant_id in participant_videos:
            participant_labels[participant_id] = intoxicated_flag

    participants = list(participant_videos.keys())
    labels = [participant_labels[p] for p in participants]

    log(f"\n{'='*50}")
    log(f"Total participants      : {len(participants)}")
    log(f"Intoxicated participants: {sum(labels)}")
    log(f"Sober participants      : {len(labels) - sum(labels)}")

    for pid in participants:
        vids = participant_videos[pid]
        log(f"  {pid}: {len(vids)} videos  label={participant_labels[pid]}", also_print=False)

    if len(participants) == 0:
        raise RuntimeError("No participants found — check FLOW_ROOT path.")

    return participant_videos, participant_labels, participants, labels

# =========================================================
# DATASET  — with .npy pre-caching into RAM
# =========================================================
class FlowOnlyDataset(Dataset):

    def __init__(self, samples, t=T, train=True, inference=False, preload=True):
        self.samples   = samples
        self.t         = t
        self.train     = train
        self.inference = inference

        # Cache sorted file lists
        self.flow_files_cache = [
            sorted(flow_video_dir.glob("*.npy"))
            for flow_video_dir, _, _ in samples
        ]

        # Optional: preload all npy arrays into RAM (fast if RAM allows)
        self.preloaded = None
        if preload:
            self.preloaded = []
            for files in tqdm(self.flow_files_cache, desc="  Preloading npy→RAM", leave=False):
                self.preloaded.append(np.stack([np.load(str(f)) for f in files]))  # (N,H,W,2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, label = self.samples[idx]

        if self.inference:
            n_frames = len(self.flow_files_cache[idx])
            start    = max(0, (n_frames - INF_T) // 2)
            t        = min(INF_T, n_frames)
        else:
            n_frames = len(self.flow_files_cache[idx])
            t        = self.t
            start    = random.randint(0, n_frames - t) if self.train else (n_frames - t) // 2

        if self.preloaded is not None:
            flows = self.preloaded[idx][start:start + t]          # (T,H,W,2)
        else:
            files = self.flow_files_cache[idx]
            flows = np.stack([np.load(str(f)) for f in files[start:start + t]])

        # (T,H,W,2) → (2,T,H,W)
        flows = np.transpose(flows, (3, 0, 1, 2)).astype(np.float32)

        return torch.from_numpy(flows), torch.tensor(label, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class FlowBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class FlowOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_branch = FlowBranch()
        self.classifier  = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, flow):
        return self.classifier(self.flow_branch(flow)).view(-1)

# =========================================================
# DATALOADER FACTORY
# =========================================================
def make_loader(samples, shuffle=True, train=True, inference=False, preload=True):
    dataset = FlowOnlyDataset(samples, train=train, inference=inference, preload=preload)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        prefetch_factor=PREFETCH if NUM_WORKERS > 0 else None,
        persistent_workers=(NUM_WORKERS > 0),
    )

# =========================================================
# TRAIN / EVAL EPOCH
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None, training=True, epoch_label=""):
    model.train() if training else model.eval()

    total_loss = 0.0
    probs_all  = []
    gts_all    = []
    t0         = time.time()

    pbar = tqdm(loader, desc=epoch_label, leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (flow, y) in enumerate(pbar):
            flow = flow.to(DEVICE, non_blocking=True)
            y    = y.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                logits = model(flow)
                loss   = criterion(logits, y)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
            gts_all.extend(y.cpu().numpy().tolist())

            # Live batch stats in tqdm postfix
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             batch=f"{batch_idx+1}/{len(loader)}")

    elapsed   = time.time() - t0
    avg_loss  = total_loss / max(len(loader), 1)

    if 0 < sum(gts_all) < len(gts_all):
        auc          = roc_auc_score(gts_all, probs_all)
        fpr, tpr, thr = roc_curve(gts_all, probs_all)
        best         = np.argmax(tpr - fpr)
        thresh       = thr[best]
        preds        = (np.array(probs_all) > thresh).astype(int)
        f1           = f1_score(gts_all, preds, zero_division=0)
        uar          = recall_score(gts_all, preds, average="macro")
    else:
        auc, f1, uar, thresh = float("nan"), 0.0, 0.0, 0.5

    return avg_loss, f1, auc, uar, thresh, elapsed

# =========================================================
# CROSS-VALIDATION
# =========================================================
def main():
    log(f"\n{'='*60}")
    log(f"DEVICE      : {DEVICE}")
    log(f"T           : {T}")
    log(f"BATCH_SIZE  : {BATCH_SIZE}")
    log(f"NUM_WORKERS : {NUM_WORKERS}")
    log(f"EPOCHS      : {EPOCHS}")
    log(f"LR          : {LR}")
    log(f"{'='*60}\n")

    participant_videos, participant_labels, participants, labels = organize_data()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(participants, labels)):

        log(f"\n{'='*60}")
        log(f"FOLD {fold+1}/5")
        log(f"{'='*60}")

        fold_dir = SAVE_ROOT / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)

        tr_parts  = [participants[i] for i in tr_idx]
        val_parts = [participants[i] for i in val_idx]

        log(f"Train participants ({len(tr_parts)}): {tr_parts}")
        log(f"Val   participants ({len(val_parts)}): {val_parts}")

        tr_samples  = [v for p in tr_parts  for v in participant_videos[p]]
        val_samples = [v for p in val_parts for v in participant_videos[p]]

        log(f"Train samples: {len(tr_samples)}  |  Val samples: {len(val_samples)}")

        tr_labels_list = [lbl for (_, _, lbl) in tr_samples]
        pos = sum(tr_labels_list)
        neg = len(tr_labels_list) - pos
        log(f"Train class dist — pos: {pos}  neg: {neg}")

        log("\nLoading train data...")
        train_loader = make_loader(tr_samples, shuffle=True,  train=True,  preload=True)
        log("Loading val data...")
        val_loader   = make_loader(val_samples, shuffle=False, train=False, preload=True)

        model = FlowOnlyModel().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"Model params: {n_params:,}")

        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / pos]).to(DEVICE)
            criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            log(f"pos_weight: {pos_weight.item():.3f}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc       = 0.0
        best_epoch     = 0
        fold_train_log = []
        fold_val_log   = []

        for epoch in range(EPOCHS):
            ep_label = f"Fold {fold+1} Ep {epoch+1}/{EPOCHS}"

            tr_loss, tr_f1, tr_auc, tr_uar, _, tr_t = run_epoch(
                model, train_loader, criterion, optimizer,
                training=True, epoch_label=f"[Train] {ep_label}"
            )
            vl_loss, vl_f1, vl_auc, vl_uar, th, vl_t = run_epoch(
                model, val_loader, criterion,
                training=False, epoch_label=f"[Val]   {ep_label}"
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            tr_line = (f"[Train] Ep{epoch+1:02d} | loss={tr_loss:.4f} "
                       f"f1={tr_f1:.4f} auc={tr_auc:.4f} uar={tr_uar:.4f} | {tr_t:.1f}s")
            vl_line = (f"[Val]   Ep{epoch+1:02d} | loss={vl_loss:.4f} "
                       f"f1={vl_f1:.4f} auc={vl_auc:.4f} uar={vl_uar:.4f} "
                       f"th={th:.3f} lr={current_lr:.2e} | {vl_t:.1f}s")

            log(tr_line)
            log(vl_line)
            fold_train_log.append(tr_line)
            fold_val_log.append(vl_line)

            torch.save(model.state_dict(), fold_dir / f"epoch_{epoch+1:02d}.pth")

            if not np.isnan(vl_auc) and vl_auc > best_auc:
                best_auc   = vl_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), fold_dir / "best_model.pth")
                log(f"  *** New best AUC: {best_auc:.4f} at epoch {best_epoch} ***")

        log(f"\nFold {fold+1} complete — Best AUC: {best_auc:.4f} (epoch {best_epoch})")
        fold_results.append(best_auc)

    log(f"\n{'='*60}")
    log("FINAL CROSS-VALIDATION RESULTS")
    log(f"{'='*60}")
    for i, a in enumerate(fold_results, 1):
        log(f"  Fold {i}: AUC = {a:.4f}")
    log(f"  Mean AUC : {np.mean(fold_results):.4f}")
    log(f"  Std  AUC : {np.std(fold_results):.4f}")
    log(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()