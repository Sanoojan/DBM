import re
import numpy as np

log_path = "Logs/New_train_cnn_pos_weight5_no_undersampling.log"

pattern = re.compile(
    r"UAR=(\d+\.\d+)\s+sens=(\d+\.\d+)\s+spec=(\d+\.\d+)\s+AUC=(\d+\.\d+)\s+F1=(\d+\.\d+)"
)

metrics = {
    "uar": [],
    "sens": [],
    "spec": [],
    "auc": [],
    "f1": []
}

with open(log_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            uar, sens, spec, auc, f1 = map(float, match.groups())
            metrics["uar"].append(uar)
            metrics["sens"].append(sens)
            metrics["spec"].append(spec)
            metrics["auc"].append(auc)
            metrics["f1"].append(f1)

# ---- print per-fold (optional) ----
print("\nPer-fold results:")
for i in range(len(metrics["uar"])):
    print(f"Fold {i+1}: "
          f"UAR={metrics['uar'][i]:.4f}, "
          f"Sens={metrics['sens'][i]:.4f}, "
          f"Spec={metrics['spec'][i]:.4f}, "
          f"AUC={metrics['auc'][i]:.4f}, "
          f"F1={metrics['f1'][i]:.4f}")

# ---- compute mean & std ----
print("\nFinal Results (Mean ± Std):")
for k, v in metrics.items():
    v = np.array(v)
    print(f"{k.upper():<5}: {v.mean():.4f} ± {v.std():.4f}")