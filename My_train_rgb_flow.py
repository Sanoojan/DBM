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
NUM_WORKERS = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Device : {DEVICE}")
print(f"T      : {T}")
print(f"Workers: {NUM_WORKERS}")

# =========================================================
# ORGANIZE DATA (Participant-Level)
# =========================================================
participant_videos = defaultdict(list)
participant_labels = {}

for participant_dir in sorted(FLOW_ROOT.iterdir()):
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

        rgb_video_dir = RGB_ROOT / participant_id / video_name
        if not rgb_video_dir.exists():
            continue

        flow_files = sorted(flow_video_dir.glob("*.npy"))
        rgb_files  = sorted(rgb_video_dir.glob("*.jpg"))

        if len(flow_files) < T:
            continue
        if len(rgb_files) < T + 1:
            continue

        participant_videos[participant_id].append(
            (flow_video_dir, video_name, label)
        )

    if participant_id in participant_videos:
        participant_labels[participant_id] = intoxicated_flag

participants = list(participant_videos.keys())
labels = [participant_labels[p] for p in participants]

print(f"\nTotal participants      : {len(participants)}")
print(f"Intoxicated participants: {sum(labels)}")

if len(participants) == 0:
    raise RuntimeError("No participants found.")

# =========================================================
# DATASET
# =========================================================
class TwoStreamDataset(Dataset):

    def __init__(self, samples, t=T, train=True):
        self.samples = samples
        self.t = t
        self.train = train

        self.flow_files_cache = []
        self.rgb_files_cache = []

        print(f"  [Dataset] Caching {len(samples)} samples...")
        for flow_video_dir, video_name, _ in samples:
            rgb_video_dir = RGB_ROOT / flow_video_dir.parent.name / video_name
            self.flow_files_cache.append(sorted(flow_video_dir.glob("*.npy")))
            self.rgb_files_cache.append(sorted(rgb_video_dir.glob("*.jpg")))
        print("  [Dataset] Done caching.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_video_dir, video_name, label = self.samples[idx]
        flow_files = self.flow_files_cache[idx]
        rgb_files  = self.rgb_files_cache[idx]

        if self.train:
            start = random.randint(0, len(flow_files) - self.t)
        else:
            start = (len(flow_files) - self.t) // 2

        # Flow clip
        flows = np.stack([
            np.load(str(f)) for f in flow_files[start:start+self.t]
        ])  # (T,H,W,2)

        flows = np.transpose(flows, (3,0,1,2)).astype(np.float32)

        # RGB middle frame
        mid = start + self.t // 2
        rgb = cv2.imread(str(rgb_files[mid]))
        rgb = cv2.resize(rgb, (224,224))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2,0,1))

        return (
            torch.from_numpy(flows),
            torch.from_numpy(rgb),
            torch.tensor(label, dtype=torch.float32)
        )

# =========================================================
# MODEL
# =========================================================
class FlowBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2,16,3,padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16,32,3,padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32,64,3,padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self,x):
        return self.features(x).view(x.size(0), -1)

class TwoStreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_branch = FlowBranch()

        self.rgb_branch = models.resnet18(weights="IMAGENET1K_V1")
        self.rgb_branch.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(64+512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )

    def forward(self,flow,rgb):
        f_feat = self.flow_branch(flow)
        r_feat = self.rgb_branch(rgb)
        fused = torch.cat([f_feat,r_feat],dim=1)
        return self.classifier(fused).view(-1)

# =========================================================
# TRAIN / VAL
# =========================================================
def make_loader(samples, shuffle, train):
    dataset = TwoStreamDataset(samples, train=train)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE=="cuda"),
        persistent_workers=False
    )

def run_epoch(model, loader, criterion, optimizer=None, training=True):

    model.train() if training else model.eval()
    total_loss = 0

    probs_all = []
    gts_all = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for flow, rgb, y in loader:

            flow = flow.to(DEVICE)
            rgb  = rgb.to(DEVICE)
            y    = y.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(flow,rgb)
            loss = criterion(logits,y)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            gts_all.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)

    if sum(gts_all)>0 and sum(gts_all)<len(gts_all):

        auc = roc_auc_score(gts_all, probs_all)

        fpr,tpr,thr = roc_curve(gts_all, probs_all)
        best = np.argmax(tpr - fpr)
        thresh = thr[best]

        preds = (np.array(probs_all)>thresh).astype(int)
        f1 = f1_score(gts_all,preds,zero_division=0)
        uar = recall_score(gts_all,preds,average="macro")

    else:
        auc,f1,uar,thresh = float("nan"),0,0,0.5

    return avg_loss,f1,auc,uar,thresh

# =========================================================
# CROSS VALIDATION
# =========================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results=[]

for fold,(tr_idx,val_idx) in enumerate(skf.split(participants,labels)):

    print(f"\n================ Fold {fold+1} ================")

    fold_dir = SAVE_ROOT / f"fold_{fold+1}"
    fold_dir.mkdir(exist_ok=True)

    tr_parts = [participants[i] for i in tr_idx]
    val_parts= [participants[i] for i in val_idx]

    tr_samples = [v for p in tr_parts for v in participant_videos[p]]
    val_samples= [v for p in val_parts for v in participant_videos[p]]

    train_loader = make_loader(tr_samples,True,True)
    val_loader   = make_loader(val_samples,False,False)

    model = TwoStreamModel().to(DEVICE)

    tr_labels_list = [lbl for (_,_,lbl) in tr_samples]
    pos = sum(tr_labels_list)
    neg = len(tr_labels_list)-pos

    if pos>0 and neg>0:
        pos_weight = torch.tensor([neg/pos]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"pos_weight = {pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_auc=0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        tr_loss,tr_f1,tr_auc,tr_uar,_ = run_epoch(
            model,train_loader,criterion,optimizer,True
        )

        print(f"[Train] loss={tr_loss:.4f} f1={tr_f1:.4f} auc={tr_auc:.4f} uar={tr_uar:.4f}")

        vl_loss,vl_f1,vl_auc,vl_uar,th = run_epoch(
            model,val_loader,criterion,None,False
        )

        print(f"[Val]   loss={vl_loss:.4f} f1={vl_f1:.4f} auc={vl_auc:.4f} uar={vl_uar:.4f} th={th:.3f}")

        torch.save(model.state_dict(), fold_dir / f"epoch_{epoch+1}.pth")

        if not np.isnan(vl_auc) and vl_auc>best_auc:
            best_auc=vl_auc
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

    fold_results.append(best_auc)
    print(f"Best AUC Fold {fold+1}: {best_auc:.4f}")

print("\n================ FINAL RESULTS ================")
for i,a in enumerate(fold_results,1):
    print(f"Fold {i}: {a:.4f}")
print(f"Mean AUC: {np.mean(fold_results):.4f}")
print(f"Std  AUC: {np.std(fold_results):.4f}")