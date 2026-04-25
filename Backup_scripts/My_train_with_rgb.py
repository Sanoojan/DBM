import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2

# =========================================================
# CONFIG
# =========================================================
FLOW_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/optical_flow_224h")
RGB_ROOT  = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/frames_resized_224h")
SAVE_ROOT = Path("./two_stream_models")
SAVE_ROOT.mkdir(exist_ok=True)

T          = 200
BATCH_SIZE = 4
EPOCHS     = 10
LR         = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

# ── Start with 0 to rule out DataLoader/fork deadlock.
# ── Once the run completes end-to-end, try 2 or 4.
NUM_WORKERS = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Device : {DEVICE}")
print(f"T      : {T}")
print(f"Workers: {NUM_WORKERS}")

# =========================================================
# ORGANIZE DATA  (Participant-Level)
# =========================================================
participant_videos = defaultdict(list)
participant_labels = {}

for participant_dir in sorted(FLOW_ROOT.iterdir()):
    if not participant_dir.is_dir():
        continue

    participant_id      = participant_dir.name
    intoxicated_flag    = 0

    for flow_video_dir in sorted(participant_dir.iterdir()):
        if not flow_video_dir.is_dir():
            continue

        video_name = flow_video_dir.name

        # Label logic
        if participant_id.startswith("P") and video_name.startswith("R2"):
            label            = 1
            intoxicated_flag = 1
        else:
            label = 0

        # Matching RGB folder
        rgb_video_dir = RGB_ROOT / participant_id / video_name
        if not rgb_video_dir.exists():
            print(f"  [SKIP] RGB folder missing: {rgb_video_dir}")
            continue

        flow_files = sorted(flow_video_dir.glob("*.npy"))
        rgb_files  = sorted(rgb_video_dir.glob("*.jpg"))

        if len(flow_files) < T:
            print(f"  [SKIP] Not enough flow frames ({len(flow_files)}<{T}): {flow_video_dir}")
            continue
        if len(rgb_files) < T + 1:
            print(f"  [SKIP] Not enough RGB frames ({len(rgb_files)}<{T+1}): {rgb_video_dir}")
            continue

        participant_videos[participant_id].append((flow_video_dir, video_name, label))

    if participant_id in participant_videos:
        participant_labels[participant_id] = intoxicated_flag

participants = list(participant_videos.keys())
labels       = [participant_labels[p] for p in participants]

print(f"\nTotal participants      : {len(participants)}")
print(f"Intoxicated participants: {sum(labels)}")

if len(participants) == 0:
    raise RuntimeError("No participants found. Check FLOW_ROOT / RGB_ROOT paths.")

# =========================================================
# DATASET
# =========================================================
class TwoStreamDataset(Dataset):
    """
    Pre-caches sorted file lists at construction time so that
    glob() is never called inside __getitem__ (which runs in
    worker processes and can cause forking issues / slowdowns).
    """

    def __init__(self, samples, t=T):
        self.samples = samples
        self.t       = t

        # Pre-cache file lists
        self.flow_files_cache: list[list[Path]] = []
        self.rgb_files_cache:  list[list[Path]] = []

        print(f"  [Dataset] Caching file lists for {len(samples)} samples ...")
        for i, (flow_video_dir, video_name, _label) in enumerate(samples):
            rgb_video_dir = RGB_ROOT / flow_video_dir.parent.name / video_name
            self.flow_files_cache.append(sorted(flow_video_dir.glob("*.npy")))
            self.rgb_files_cache.append(sorted(rgb_video_dir.glob("*.jpg")))
            if (i + 1) % 20 == 0 or (i + 1) == len(samples):
                print(f"    {i+1}/{len(samples)}", flush=True)
        print("  [Dataset] Caching complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _flow_video_dir, _video_name, label = self.samples[idx]

        flow_files = self.flow_files_cache[idx]
        rgb_files  = self.rgb_files_cache[idx]

        t = self.t
        start = random.randint(0, len(flow_files) - t)

        # ── Flow: stack T consecutive .npy files → (2, T, H, W)
        flows = np.stack([
            np.load(str(f)) for f in flow_files[start : start + t]
        ])                                              # (T, H, W, 2)
        flows = np.transpose(flows, (3, 0, 1, 2)).astype(np.float32)  # (2,T,H,W)

        # ── RGB: single middle frame → (3, 224, 224)
        mid_idx = start + t // 2
        rgb = cv2.imread(str(rgb_files[mid_idx]))
        if rgb is None:
            raise FileNotFoundError(f"cv2 could not read: {rgb_files[mid_idx]}")
        rgb = cv2.resize(rgb, (224, 224))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))              # (3, 224, 224)

        return (
            torch.from_numpy(flows),
            torch.from_numpy(rgb),
            torch.tensor(label, dtype=torch.float32),
        )

# =========================================================
# MODEL
# =========================================================
class FlowBranch(nn.Module):
    """Lightweight 3-D CNN for optical flow clips (2, T, H, W)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),          # → (B, 64, 1, 1, 1)
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)   # (B, 64)


class TwoStreamModel(nn.Module):
    """Fuses optical-flow branch (64-d) with ResNet-18 RGB branch (512-d)."""

    def __init__(self):
        super().__init__()
        self.flow_branch = FlowBranch()

        self.rgb_branch = models.resnet18(weights="IMAGENET1K_V1")
        self.rgb_branch.fc = nn.Identity()             # output: 512-d

        self.classifier = nn.Sequential(
            nn.Linear(64 + 512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, flow, rgb):
        flow_feat = self.flow_branch(flow)             # (B, 64)
        rgb_feat  = self.rgb_branch(rgb)               # (B, 512)
        fused     = torch.cat([flow_feat, rgb_feat], dim=1)
        return self.classifier(fused).view(-1)         # (B,)

# =========================================================
# HELPERS
# =========================================================
def make_loader(samples, shuffle, num_workers=NUM_WORKERS):
    dataset = TwoStreamDataset(samples)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda") and (num_workers == 0),
        persistent_workers=False,   # must be False when num_workers==0
    )


def run_epoch(model, loader, criterion, optimizer, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    preds_list, gts_list, probs_list = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (flow, rgb, y) in enumerate(loader):
            flow = flow.to(DEVICE, non_blocking=True)
            rgb  = rgb.to(DEVICE, non_blocking=True)
            y    = y.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(flow, rgb)
            loss   = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob > 0.5).astype(int)
            probs_list.extend(prob.tolist())
            preds_list.extend(pred.tolist())
            gts_list.extend(y.cpu().numpy().tolist())

            print(
                f"    {'Train' if training else 'Val'} "
                f"batch {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}",
                end="\r",
                flush=True,
            )

    print()  # newline after \r progress

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(gts_list, preds_list)
    f1       = f1_score(gts_list, preds_list, zero_division=0)

    n_pos = sum(gts_list)
    n_neg = len(gts_list) - n_pos
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(gts_list, probs_list)
    else:
        auc = float("nan")
        print("    [WARN] Only one class in batch — AUC undefined.")

    return avg_loss, acc, f1, auc

# =========================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# =========================================================
skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = []   # best val AUC per fold

for fold, (train_idx, val_idx) in enumerate(skf.split(participants, labels)):

    print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")
    fold_dir = SAVE_ROOT / f"fold_{fold+1}"
    fold_dir.mkdir(exist_ok=True)

    train_participants = [participants[i] for i in train_idx]
    val_participants   = [participants[i] for i in val_idx]

    train_samples = [v for p in train_participants for v in participant_videos[p]]
    val_samples   = [v for p in val_participants   for v in participant_videos[p]]

    print(f"  Train samples: {len(train_samples)}  |  Val samples: {len(val_samples)}")

    # Class-weighted loss
    train_labels_list = [lbl for (_, _, lbl) in train_samples]
    pos_count = sum(train_labels_list)
    neg_count = len(train_labels_list) - pos_count

    if pos_count == 0 or neg_count == 0:
        print("  [WARN] Only one class in training fold — using unweighted loss.")
        criterion = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  pos_weight = {pos_weight.item():.3f}  "
              f"(pos={pos_count}, neg={neg_count})")

    print("\n  Building train loader ...")
    train_loader = make_loader(train_samples, shuffle=True)
    print("  Building val loader ...")
    val_loader   = make_loader(val_samples,   shuffle=False)

    model     = TwoStreamModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_auc  = 0.0

    for epoch in range(EPOCHS):
        print(f"\n  -- Epoch {epoch+1}/{EPOCHS} --")

        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, training=True
        )
        print(f"  [Train] loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"f1={tr_f1:.4f}  auc={tr_auc:.4f}")

        vl_loss, vl_acc, vl_f1, vl_auc = run_epoch(
            model, val_loader, criterion, optimizer=None, training=False
        )
        print(f"  [Val]   loss={vl_loss:.4f}  acc={vl_acc:.4f}  "
              f"f1={vl_f1:.4f}  auc={vl_auc:.4f}")

        if not np.isnan(vl_auc) and vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            print(f"  ✅  Best model updated  (AUC={best_auc:.4f})")

    fold_results.append(best_auc)
    print(f"\n  Fold {fold+1} best AUC: {best_auc:.4f}")

# =========================================================
# FINAL RESULTS
# =========================================================
print("\n" + "="*50)
print("FINAL CROSS-VALIDATION RESULTS")
print("="*50)
for i, auc in enumerate(fold_results, 1):
    print(f"  Fold {i}: AUC = {auc:.4f}")
print(f"  Mean AUC : {np.mean(fold_results):.4f}")
print(f"  Std  AUC : {np.std(fold_results):.4f}")