import argparse
import csv
import random
import numpy as np
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from models.models_all import *

from sklearn.metrics import roc_curve


parser = argparse.ArgumentParser()

parser.add_argument("--fold", type=int, default=None)

parser.add_argument("--model", type=str, default="VideoModel",

                    help="Model type: r3d18, CustomR3D18, VideoModel, etc.")

parser.add_argument("--backbone", type=str, default="r3d18",

                    help="Backbone for VideoModel: r3d18, r2plus1d_18, mc3_18, resnet18, vit_b_16")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")

args = parser.parse_args()



print(f"Using model: {args.model}  backbone: {args.backbone}")
print(f"Batch size: {args.batch_size}")


FLOW_HDF5_DIR = Path("dataset/Vehicle/flow_hdf5_frame_chunks_raft_large")
FOLDS_CSV     = Path("hail-datasets/hail_datasets/datasets/ddd_2024/folds.csv")
TEST_ONLY = False  # set to True to skip training and only run test (requires existing checkpoints)

WANDB_PROJECT = "DBM-OFlow"  # set to None to disable

EXCLUDE_PARTICIPANTS = {"P701", "P711", "7218", "7219", "7225", "7228", "7229", "7237"}
SCENARIO_NAMES       = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]

MODEL = args.model  # "cnn" or "r3d18"
MODEL_NAME = args.backbone  # for logging; should be consistent with build_model()
NAME = f"{MODEL}_{MODEL_NAME}_cropped"  # for saving logs and models; include key config in the name
CHUNK_FRAMES    = 300
BUFFER_FRAMES   = 300
VAL_CHUNKS      = 4
CHUNK_STRIDE    = 300
TEMPORAL_STRIDE = 2    # r3d18: use 2 (→150 frames); cnn: 1

BATCH_SIZE  = args.batch_size       # r3d18 is heavier; use 8 for cnn
EPOCHS      = 20
LR          = 1e-5 
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
NUM_WORKERS = 4

SAVE_ROOT = Path(f"./Trained_checkpoints/models_{NAME}_{MODEL}_{MODEL_NAME}_c{CHUNK_FRAMES}_b{BUFFER_FRAMES}_ts{TEMPORAL_STRIDE}_ep{EPOCHS}_lr{LR}_bs{BATCH_SIZE}")
SAVE_ROOT.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

LOG_FILE = SAVE_ROOT / "training_log.txt"


def find_best_threshold(gts, probs):

    fpr, tpr, thresholds = roc_curve(gts, probs)

    uar_scores = (tpr + (1 - fpr)) / 2

    best_idx = np.argmax(uar_scores)

    return thresholds[best_idx]

def topk_continuity_loss(frame_logits, k_ratio=0.1):

    B, T = frame_logits.shape

    k = max(1, int(T * k_ratio))

    topk_vals, topk_idx = torch.topk(frame_logits, k=k, dim=1)

    # sort indices

    topk_idx_sorted, _ = torch.sort(topk_idx, dim=1)

    # distance between consecutive top frames

    diffs = topk_idx_sorted[:, 1:] - topk_idx_sorted[:, :-1]

    # penalize large gaps

    return diffs.float().mean()

def topk_scores(frame_logits, k_ratio=0.1):

    B, T = frame_logits.shape

    k = max(1, int(T * k_ratio))

    topk_vals, _ = torch.topk(frame_logits, k=k, dim=1)

    return topk_vals.mean(dim=1)  # (B,)

def ranking_loss(frame_logits, labels, margin=1.0, k_ratio=0.1):

    # frame_logits: (B, T)

    # labels: (B,)

    scores = topk_scores(frame_logits, k_ratio)  # (B,)

    pos_scores = scores[labels == 1]

    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:

        return torch.tensor(0.0, device=frame_logits.device)

    # pairwise ranking

    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (Np, Nn)

    loss = torch.clamp(margin - diff, min=0).mean()

    return loss

def ranking_loss_with_std(frame_logits, labels, margin=1.0, k_ratio=0.1, lambda_std=0.1):

    # ---- top-k scores ----

    scores = topk_scores(frame_logits, k_ratio)  # (B,)

    pos_scores = scores[labels == 1]

    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:

        rank_loss = torch.tensor(0.0, device=frame_logits.device)

    else:

        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        rank_loss = torch.clamp(margin - diff, min=0).mean()

    # ---- std term ----

    std = frame_logits.std(dim=1)  # (B,)

    pos_std = std[labels == 1]

    neg_std = std[labels == 0]

    std_loss = 0.0

    if len(pos_std) > 0:

        std_loss += -pos_std.mean()   # maximize std for positives

    if len(neg_std) > 0:

        std_loss += neg_std.mean()    # minimize std for negatives

    return rank_loss + lambda_std * std_loss

def mil_topk(frame_logits, k_ratio=0.1):

    # frame_logits: (B, T)

    B, T = frame_logits.shape

    k = max(1, int(T * k_ratio))

    topk_vals, _ = torch.topk(frame_logits, k=k, dim=1)

    video_logits = topk_vals.mean(dim=1)  # (B,)

    return video_logits

def build_model():
    if MODEL == "r3d18":
        return build_r3d18()
    elif MODEL == "CustomR3D18":
        return CustomR3D18()
    
    elif MODEL == "FrameTemporalModel":
        return FrameTemporalModel()
    
    elif MODEL == "MILFrameModel": 
        return MILFrameModel()
    
    elif MODEL == "MILFrameTransformer":
        return MILFrameTransformer()
    elif MODEL == "VideoModel":
        return VideoModel(backbone_name=MODEL_NAME,crop=True)

    else:
        return SmallCNN()

def attention_entropy(attn):

    # attn: (B, T)

    return -(attn * torch.log(attn + 1e-8)).sum(dim=1).mean()

def topk_mean(logits, k_ratio=0.1):

    # logits: (B, T)

    B, T = logits.shape

    k = max(1, int(T * k_ratio))

    topk_vals, _ = torch.topk(logits, k=k, dim=1)

    return topk_vals.mean(dim=1)  # (B,)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def load_folds(folds_csv):
    participant_fold = {}
    with open(folds_csv) as f:
        for fold_idx, line in enumerate(csv.reader(f)):
            for pid in line:
                pid = pid.strip()
                if pid:
                    participant_fold[pid] = fold_idx
    return participant_fold


def scan_data(flow_hdf5_dir, participant_fold):
    import time
    samples    = []
    min_frames = CHUNK_FRAMES + 2 * BUFFER_FRAMES

    for h5_path in sorted(flow_hdf5_dir.glob("*_flow.h5")):
        pid = h5_path.stem[:-5]
        if pid in EXCLUDE_PARTICIPANTS or pid not in participant_fold:
            continue
        fold = participant_fold[pid]

        t0 = time.perf_counter()
        with h5py.File(h5_path, "r") as hf:
            for key in hf.keys():
                if "practice" in key:
                    continue
                n_frames = hf[key]["flow"].shape[0]
                label    = int(hf[key]["label"][()])
                if n_frames < min_frames:
                    continue
                n_train_chunks = max(1, (n_frames - 2 * BUFFER_FRAMES) // CHUNK_FRAMES)
                samples.append({
                    "participant":    pid,
                    "fold":           fold,
                    "label":          label,
                    "h5_path":        h5_path,
                    "key":            key,
                    "n_frames":       n_frames,
                    "n_train_chunks": n_train_chunks,
                })
        print(f"  scan {pid}: {time.perf_counter()-t0:.2f}s  "
              f"({len([s for s in samples if s['participant']==pid])} sequences)")

    print(f"scan_data done: {len(samples)} sequences total")
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
                    self.items.append((s["h5_path"], s["key"], s["n_frames"], None, s["label"]))
            else:
                for start in get_val_starts(s["n_frames"]):
                    self.items.append((s["h5_path"], s["key"], s["n_frames"], start, s["label"]))
        self._handles = {}

    def _get_handle(self, h5_path):
        key = str(h5_path)
        if key not in self._handles:
            self._handles[key] = h5py.File(h5_path, "r")
        return self._handles[key]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        h5_path, key, n_frames, start, label = self.items[idx]
        if self.train:
            start = get_train_start(n_frames)
        hf   = self._get_handle(h5_path)
        flow = hf[key]["flow"][start : start + CHUNK_FRAMES : TEMPORAL_STRIDE]
        flow = flow.astype(np.float32)
        flow = np.transpose(flow, (3, 0, 1, 2))   # (2, T, H, W)
        
        flow = np.clip(flow, -20, 20) / 20.0
        return torch.from_numpy(flow), torch.tensor(label, dtype=torch.float32)




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
    return {"uar": uar, "sensitivity": sensitivity, "specificity": specificity, "auc": auc, "f1": f1}


def run_epoch(model, loader, criterion, criterion_frame, optimizer=None, training=True, desc=""):

    model.train() if training else model.eval()

    total_loss, probs_all, gts_all = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:

        pbar = tqdm(loader, desc=desc, leave=False)

        for flow, y in pbar:

            flow = flow.to(DEVICE, non_blocking=True)

            y    = y.to(DEVICE, non_blocking=True)

            

            if training:

                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):

                frame_logits=model(flow)
                if MODEL_NAME == "resnet18" or MODEL_NAME == "resnet50":
                   
                    logits = mil_topk(frame_logits, k_ratio=0.1)

                    loss_cls = criterion(logits, y)
                    loss_rank = ranking_loss_with_std(frame_logits, y)

                    loss = 0.0 * loss_cls + 0.1 * loss_rank
                else:
                    logits=frame_logits    
                
                loss_video = criterion(logits, y)
                
                # ✅ FIX: use logits directly (no sigmoid)

                # loss_frame = criterion_frame(
                #     frame_logits,
                #     y.unsqueeze(1).expand_as(frame_logits)

                # )
                # loss_cont = topk_continuity_loss(frame_logits)
                # loss_attn = attention_entropy(attn)

                # 🔥 slightly reduced frame loss weight (important)

                loss = loss_video 
                # + 0.1 * loss_frame + 0.01 * loss_attn
                # loss += 0.01 * loss_cont

            if training:

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
            total_loss += loss.item()
            probs_all.extend(torch.sigmoid(logits).detach().float().cpu().numpy().tolist())
            gts_all.extend(y.cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = max(len(loader), 1)

    # default metrics (threshold=0.5 just for logging)

    metrics = compute_metrics(gts_all, probs_all)

    return total_loss / n, metrics, gts_all, probs_all


def make_loader(samples, train):
    ds = FlowDataset(samples, train=train)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                      prefetch_factor=2 if NUM_WORKERS > 0 else None,
                      persistent_workers=(NUM_WORKERS > 0))


def main():
    import argparse, json
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--fold", type=int, default=None,
    #                     help="Run a single test fold (0-4). Omit to run all 5.")
    # args = parser.parse_args()
    folds_to_run = list(range(5)) if args.fold is None else [args.fold]

    if WANDB_PROJECT:
        import wandb as wb
        wb.init(project=WANDB_PROJECT, name=NAME, config=dict(
            model=MODEL, chunk_frames=CHUNK_FRAMES, buffer_frames=BUFFER_FRAMES,
            temporal_stride=TEMPORAL_STRIDE, batch_size=BATCH_SIZE,
            epochs=EPOCHS, lr=LR, seed=SEED, fold=args.fold,
        ))

    log(f"\n{'='*60}")
    log(f"DEVICE={DEVICE}  CHUNK={CHUNK_FRAMES}fr  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}  folds={folds_to_run}")
    log(f"{'='*60}")

    participant_fold = load_folds(FOLDS_CSV)
    all_samples      = scan_data(FLOW_HDF5_DIR, participant_fold)

    log(f"\nTotal sequences: {len(all_samples)}")
    log(f"  Intoxicated: {sum(s['label']==1 for s in all_samples)}")
    log(f"  Sober:       {sum(s['label']==0 for s in all_samples)}")

    fold_test_uars = []

    for test_fold in folds_to_run:
        val_fold = (test_fold + 1) % 5
        train_s  = [s for s in all_samples if s["fold"] != test_fold and s["fold"] != val_fold]
        val_s    = [s for s in all_samples if s["fold"] == val_fold]
        test_s   = [s for s in all_samples if s["fold"] == test_fold]

        

        intox_s = [s for s in train_s if s["label"] == 1]
        sober_s = [s for s in train_s if s["label"] == 0]
        sober_s = random.sample(sober_s, min(len(intox_s), len(sober_s)))
        train_s = intox_s + sober_s
        random.shuffle(train_s)

        log(f"\n{'='*60}")
        log(f"FOLD {test_fold+1}/5  (test={test_fold}, val={val_fold})")
        log(f"  Train: {len(train_s)} ({len(intox_s)} intox + {len(sober_s)} sober)  "
            f"Val: {len(val_s)}  Test: {len(test_s)}")

        train_loader = make_loader(train_s, train=True)
        val_loader   = make_loader(val_s,   train=False)
        test_loader  = make_loader(test_s,  train=False)

        model     = build_model().to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        criterion_frame = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        fold_dir     = SAVE_ROOT / f"fold_{test_fold+1}"
        fold_dir.mkdir(exist_ok=True)
        best_val_uar = -1.0

        best_epoch   = 0

        best_thresh  = 0.5  # store best threshold
        if not TEST_ONLY:
            for epoch in range(EPOCHS):

                tr_loss, tr_m, _, _ = run_epoch(
                    model, train_loader, criterion, criterion_frame, optimizer,
                    training=True, desc=f"F{test_fold+1} Ep{epoch+1} train"
                )

                vl_loss, vl_m, val_gts, val_probs = run_epoch(
                    model, val_loader, criterion, criterion_frame,
                    training=False, desc=f"F{test_fold+1} Ep{epoch+1} val"
                )

                scheduler.step()

                # ✅ compute best threshold on validation

                val_thresh = find_best_threshold(val_gts, val_probs)

                # ✅ recompute metrics using best threshold

                vl_m_thresh = compute_metrics(val_gts, val_probs, threshold=val_thresh)

                log(f"  Ep{epoch+1:02d} | train loss={tr_loss:.4f} uar={tr_m['uar']:.4f} auc={tr_m['auc']:.4f} | "
                    f"val loss={vl_loss:.4f} uar={vl_m_thresh['uar']:.4f} "
                    f"sens={vl_m_thresh['sensitivity']:.4f} spec={vl_m_thresh['specificity']:.4f} "
                    f"auc={vl_m_thresh['auc']:.4f} thr={val_thresh:.3f}")

                if WANDB_PROJECT:

                    wb.log({

                        f"fold": test_fold + 1,
                        f"epoch": epoch + 1,
                        f"train/{test_fold+1}_loss": tr_loss,
                        f"train/{test_fold+1}_uar": tr_m["uar"],
                        f"train/{test_fold+1}_auc": tr_m["auc"],
                        f"val/{test_fold+1}_loss": vl_loss,
                        f"val/{test_fold+1}_uar": vl_m_thresh["uar"],
                        f"val/{test_fold+1}_sensitivity": vl_m_thresh["sensitivity"],
                        f"val/{test_fold+1}_specificity": vl_m_thresh["specificity"],
                        f"val/{test_fold+1}_auc": vl_m_thresh["auc"],
                        f"val/{test_fold+1}_f1": vl_m_thresh["f1"],
                        f"val/{test_fold+1}_threshold": val_thresh

                    })

                torch.save(model.state_dict(), fold_dir / f"epoch_{epoch+1:02d}.pth")

                uar_val = vl_m_thresh["uar"] if not np.isnan(vl_m_thresh["uar"]) else -1.0

                if uar_val > best_val_uar:
                    best_val_uar = uar_val
                    best_epoch   = epoch + 1
                    best_thresh  = val_thresh   # ✅ save threshold
                    torch.save(model.state_dict(), fold_dir / "best_model.pth")

                    log(f"    *** new best val UAR={best_val_uar:.4f} (epoch {best_epoch}) thr={best_thresh:.3f} ***")

        model.load_state_dict(torch.load(fold_dir / "best_model.pth"))

        _, _, test_gts, test_probs = run_epoch(

            model, test_loader, criterion, criterion_frame,
            training=False, desc=f"F{test_fold+1} test"
        )

        # ✅ apply validation threshold

        best_thresh= best_thresh if not np.isnan(best_thresh) else 0.5
        test_m = compute_metrics(test_gts, test_probs, threshold=best_thresh)

        log(f"\n  FOLD {test_fold+1} TEST (best epoch={best_epoch}):")
        log(f"    UAR={test_m['uar']:.4f}  sens={test_m['sensitivity']:.4f}  "
            f"spec={test_m['specificity']:.4f}  AUC={test_m['auc']:.4f}  "
            f"F1={test_m['f1']:.4f}  thr={best_thresh:.3f}")
        
        if WANDB_PROJECT:
            wb.log({f"test/fold{test_fold+1}_uar": test_m["uar"],
                    f"test/fold{test_fold+1}_auc": test_m["auc"],
                    f"test/fold{test_fold+1}_sensitivity": test_m["sensitivity"],
                    f"test/fold{test_fold+1}_specificity": test_m["specificity"],
                    f"test/fold{test_fold+1}_f1": test_m["f1"]})
        fold_test_uars.append(test_m["uar"])

        result = {"fold": test_fold, "best_epoch": best_epoch, **test_m}
        with open(fold_dir / "test_results.json", "w") as f:
            json.dump(result, f, indent=2)

    if len(folds_to_run) > 1:
        log(f"\n{'='*60}")
        log("FINAL 5-FOLD TEST RESULTS")
        for fold_idx, uar in zip(folds_to_run, fold_test_uars):
            log(f"  Fold {fold_idx+1}: UAR={uar:.4f}")
        log(f"  Mean UAR: {np.mean(fold_test_uars):.4f}  Std: {np.std(fold_test_uars):.4f}")
        if WANDB_PROJECT:
            wb.log({"test/mean_uar": float(np.mean(fold_test_uars)),
                    "test/std_uar":  float(np.std(fold_test_uars))})

    if WANDB_PROJECT:
        wb.finish()


if __name__ == "__main__":
    main()
