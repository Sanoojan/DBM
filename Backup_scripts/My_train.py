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

# =========================================================
# CONFIG
# =========================================================
FLOW_ROOT = Path("/egr/research-sprintai/baliahsa/projects/DBM/dataset/Vehicle/optical_flow_224h")
SAVE_ROOT = Path("./flow_models2")
SAVE_ROOT.mkdir(exist_ok=True)

T = 200
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =========================================================
# ORGANIZE DATA (Participant-Level)
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
# DATASET
# =========================================================
class FlowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        flow_files = sorted(video_dir.glob("*.npy"))

        start = random.randint(0, len(flow_files) - T)
        selected = flow_files[start:start+T]

        flows = [np.load(f) for f in selected]
        flows = np.stack(flows)                # (T, H, W, 2)
        flows = np.transpose(flows, (3,0,1,2)) # (2, T, H, W)

        return (
            torch.tensor(flows, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

# =========================================================
# MODEL
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
        return x.view(-1)   # ALWAYS returns shape [B]

# =========================================================
# 5-FOLD STRATIFIED CV
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(participants, labels)):

    print(f"\n================ Fold {fold+1} ================")

    fold_dir = SAVE_ROOT / f"fold_{fold+1}"
    fold_dir.mkdir(exist_ok=True)

    train_participants = [participants[i] for i in train_idx]
    val_participants   = [participants[i] for i in val_idx]

    train_samples = []
    val_samples = []

    for p in train_participants:
        train_samples.extend(participant_videos[p])

    for p in val_participants:
        val_samples.extend(participant_videos[p])

    # ----------------------------
    # Compute pos_weight
    # ----------------------------
    train_labels = [label for (_, label) in train_samples]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count

    print(f"Train positives: {pos_count}")
    print(f"Train negatives: {neg_count}")

    if pos_count == 0:
        criterion = nn.BCEWithLogitsLoss()
        print("⚠ No positive samples in this fold.")
    else:
        pos_weight_value = neg_count / pos_count
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
        print(f"Using pos_weight = {pos_weight_value:.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(
        FlowDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        FlowDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Flow3DCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = 0.0

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # -------- TRAIN --------
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)          # shape [B]
            y = y.view(-1)             # ensure shape [B]

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Train Loss: {running_loss/len(train_loader):.4f}")

        # -------- VALIDATE --------
        model.eval()
        preds, gts, probs = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x)      # shape [B]
                y = y.view(-1)

                prob = torch.sigmoid(logits)

                probs.extend(prob.cpu().numpy())
                preds.extend((prob.cpu().numpy() > 0.5).astype(int))
                gts.extend(y.cpu().numpy())

        acc = accuracy_score(gts, preds)
        f1  = f1_score(gts, preds)
        auc = roc_auc_score(gts, probs)

        print(f"Val ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # -------- SAVE EVERY EPOCH --------
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "auc": auc
        }, fold_dir / f"epoch_{epoch+1}.pth")

        # -------- SAVE BEST --------
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), fold_dir / "best_model.pth")
            print("✅ Best model updated.")

    fold_results.append(best_auc)

# =========================================================
# FINAL RESULTS
# =========================================================
print("\n================ FINAL 5-FOLD RESULTS ================")
print("Fold AUCs:", fold_results)
print("Mean AUC :", np.mean(fold_results))