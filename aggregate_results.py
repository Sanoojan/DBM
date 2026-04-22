import json
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("save_root", type=str,
                    help="Path to the model save directory (e.g. models_r3d18_c300_b300_ts1_ep20_lr0.0001_bs4)")
args = parser.parse_args()

save_root = Path(args.save_root)
metrics   = ["uar", "sensitivity", "specificity", "auc", "f1"]
results   = {}

print(f"\nAggregating results from: {save_root}")
print("=" * 60)

for fold_idx in range(5):
    result_file = save_root / f"fold_{fold_idx+1}" / "test_results.json"
    if not result_file.exists():
        print(f"  Fold {fold_idx+1}: NOT FOUND ({result_file})")
        continue
    with open(result_file) as f:
        data = json.load(f)
    results[fold_idx] = data
    print(f"  Fold {fold_idx+1} (best epoch={data['best_epoch']}): "
          f"UAR={data['uar']:.4f}  sens={data['sensitivity']:.4f}  "
          f"spec={data['specificity']:.4f}  AUC={data['auc']:.4f}  F1={data['f1']:.4f}")

if not results:
    print("No results found.")
    raise SystemExit(1)

print(f"\n{'=' * 60}")
print(f"{'SUMMARY':^60}")
print(f"{'=' * 60}")
print(f"  Folds completed: {sorted(k+1 for k in results)}")
print()

for m in metrics:
    vals = [v[m] for v in results.values() if not np.isnan(v[m])]
    if vals:
        print(f"  {m:<14} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"per-fold: {[f'{v:.4f}' for v in [results[k][m] for k in sorted(results)]]}")

if len(results) == 5:
    summary = {m: {"mean": float(np.mean([v[m] for v in results.values()])),
                   "std":  float(np.std([v[m] for v in results.values()]))}
               for m in metrics}
    out = save_root / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out}")
else:
    print(f"\n  ({5 - len(results)} fold(s) still missing — run again once all complete)")
