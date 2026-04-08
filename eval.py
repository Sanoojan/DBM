import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
FLOW_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/optical_flow_224h")
MODEL_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/flow_models2")

T = 400
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# ORGANIZE DATA (SAME AS TRAINING)
# =========================================================
participant_videos = defaultdict(list)
participant_labels = {}

for participant_dir in FLOW_ROOT.iterdir():
    if not participant_dir.is_dir():
        continue

    participant_id = participant_dir.name
    intoxicated_flag = 0

    for video_dir in participant_dir.iterdir():
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name

        if participant_id.startswith("P") and video_name.startswith("R2"):
            label = 1
            intoxicated_flag = 1
        else:
            label = 0

        flow_files = sorted(video_dir.glob("*.npy"))
        if len(flow_files) >= T:
            participant_videos[participant_id].append((video_dir, label))

    if participant_id in participant_videos:
        participant_labels[participant_id] = intoxicated_flag

participants = list(participant_videos.keys())
labels = [participant_labels[p] for p in participants]

print("Total participants:", len(participants))
print("Intoxicated participants:", sum(labels))

# =========================================================
# DATASET (DETERMINISTIC CENTER CLIP)
# =========================================================
class FlowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        flow_files = sorted(video_dir.glob("*.npy"))

        # deterministic center clip
        start = (len(flow_files) - T) // 2
        selected = flow_files[start:start+T]

        flows = [np.load(f) for f in selected]
        flows = np.stack(flows)                # (T, H, W, 2)
        flows = np.transpose(flows, (3,0,1,2)) # (2, T, H, W)

        return (
            torch.tensor(flows, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# =========================================================
# MODEL (SAME ARCHITECTURE)
# =========================================================
class Flow3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1)

# =========================================================
# 5-FOLD EVALUATION
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(participants, labels)):

    print(f"\n================ Evaluating Fold {fold+1} ================")

    val_participants = [participants[i] for i in val_idx]

    val_samples = []
    for p in val_participants:
        val_samples.extend(participant_videos[p])

    val_loader = DataLoader(
        FlowDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Flow3DCNN().to(DEVICE)

    model_path = MODEL_ROOT / f"fold_{fold+1}" / "best_model.pth"
    print("Loading:", model_path)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs = []
    gts = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            prob = torch.sigmoid(logits)

            probs.extend(prob.cpu().numpy())
            gts.extend(y.cpu().numpy())

    # =====================================================
    # METRICS
    # =====================================================
    auc = roc_auc_score(gts, probs)

    fpr, tpr, thresholds = roc_curve(gts, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    preds = (np.array(probs) > optimal_threshold).astype(int)

    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds)
    uar = recall_score(gts, preds, average="macro")

    print(f"AUC  : {auc:.4f}")
    print(f"ACC  : {acc:.4f}")
    print(f"F1   : {f1:.4f}")
    print(f"UAR  : {uar:.4f}")
    print(f"Best Threshold: {optimal_threshold:.4f}")

    all_fold_results.append(auc)

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n================ FINAL RESULTS ================")
print("Fold AUCs:", all_fold_results)
print("Mean AUC :", np.mean(all_fold_results))
print("Std  AUC :", np.std(all_fold_results))