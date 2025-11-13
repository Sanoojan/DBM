#!/usr/bin/env python3
import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

total_repeats = 1


def calc_stats(gt, prob):
    stats = {}

    # UARs and AUCs
    for task_idx, task_name in enumerate(["cd", "intox", "impaired", "impaired_intox"]):
        is_valid = gt[:, task_idx] != -1
        task_gt = gt[is_valid, task_idx]
        task_prob = prob[is_valid, task_idx]
        task_pred = (task_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(task_gt, task_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        uar = (specificity + sensitivity) / 2.0
        stats[task_name + "_uar"] = uar

        # try:
        #     stats[task_name + "_auc"] = roc_auc_score(task_gt, task_prob)
        # except:
        #     stats[task_name + "_auc"] = np.nan

    # Three class confusion matrix for impairment detected first
    gt3 = gt[:, 3] + 1
    pred_bl = prob[:, 2] < 0.5
    pred3_impair_first = (prob[:, 3] >= 0.5).astype(int) + 1
    pred3_impair_first[pred_bl] = 0
    cm3 = confusion_matrix(gt3, pred3_impair_first, labels=[0, 1, 2])
    cm3_totals = np.sum(cm3, axis=1, keepdims=True).repeat(repeats=3, axis=1)
    cm3_totals[cm3_totals == 0] = 1
    cm3 = cm3 / cm3_totals
    stats["cm3_impaired_first"] = cm3.copy()

    # Three class confusion matrix us CD/intox predictions
    pred_intox = prob[:, 1] >= 0.5
    pred3_intox_cd = (prob[:, 0] >= 0.5).astype(int)
    pred3_intox_cd[pred_intox] = 2
    cm3 = confusion_matrix(gt3, pred3_intox_cd, labels=[0, 1, 2])
    cm3_totals = np.sum(cm3, axis=1, keepdims=True).repeat(repeats=3, axis=1)
    cm3_totals[cm3_totals == 0] = 1
    cm3 = cm3 / cm3_totals
    stats["cm3_intox_cd"] = cm3.copy()

    return stats


def load_checkpoints(paths, pretty_name):
    # If single path without list, put in list
    if not isinstance(paths, list):
        paths = [
            paths,
        ]

    # Loop through all repeats
    exp_stats = []
    for repeat_idx in range(total_repeats):
        # Get stats from all folds
        repeat_stats = []
        for fold_idx in range(5):
            # Loop through all paths
            fold_gt = []
            fold_prob = []
            for path in paths:
                # Find and load test pickle data
                exp_name = os.path.basename(path)
                run_name = f"{exp_name}-{fold_idx}-{repeat_idx}"
                test_path = os.path.join(path, run_name, "best_val_joint_uar", "test.pkl")
                if not os.path.isfile(test_path):
                    continue
                test_data = None
                with open(test_path, "rb") as test_file:
                    test_data = pickle.load(test_file)
                if test_data is None:
                    raise Exception(f"Couldn't load pickle file {test_path}")

                # Loop through each batch and gather predictions
                task_gt = []
                task_prob = []
                for batch_data in test_data:
                    task_gt.append(batch_data["gt"])
                    task_prob.append(batch_data["prob"])

                # Add all task data to batch
                fold_gt.append(np.concat(task_gt, axis=0))
                fold_prob.append(np.concat(task_prob, axis=0))

            # Create final fold arrays
            fold_gt = np.concat(fold_gt, axis=-1)
            fold_prob = np.concat(fold_prob, axis=-1)

            # Calc fold stats and add to repeat
            fold_stats = calc_stats(fold_gt, fold_prob)
            repeat_stats.append(fold_stats)

        # Calc meta stats across repeat
        merged_repeat_stats = {}
        for key in repeat_stats[0].keys():
            all_values = np.stack([x[key] for x in repeat_stats], axis=0)
            merged_repeat_stats[f"{key}_mean"] = np.mean(all_values, axis=0)
            merged_repeat_stats[f"{key}_std"] = np.std(all_values, axis=0)
        exp_stats.append(merged_repeat_stats)

    # Merge over all repeats
    outputs = {}
    for key in exp_stats[0].keys():
        all_values = np.stack([x[key] for x in exp_stats], axis=0)
        outputs[key] = np.mean(all_values, axis=0)

    # Print the performance row
    pm = "\u00B1"
    print(
        "{:s} & {:0.2f}{:s}{:0.2f} & {:0.2f}{:s}{:0.2f} & {:0.2f}{:s}{:0.2f} & {:0.2f}{:s}{:0.2f} \\\\".format(
            pretty_name,
            outputs["cd_uar_mean"],
            pm,
            outputs["cd_uar_std"],
            outputs["intox_uar_mean"],
            pm,
            outputs["intox_uar_std"],
            outputs["impaired_uar_mean"],
            pm,
            outputs["impaired_uar_std"],
            outputs["impaired_intox_uar_mean"],
            pm,
            outputs["impaired_intox_uar_std"],
        )
    )

    return outputs


def main():
    load_checkpoints("../checkpoints/rf", "Random Forest")
    load_checkpoints("../checkpoints/gbm", "Grad. Boosting")
    load_checkpoints("../checkpoints/svm", "SVM")
    load_checkpoints(["../checkpoints/linreg"], "Logistic Reg.")
    load_checkpoints(
        [
            "../checkpoints/shallow_cd",
            "../checkpoints/shallow_intox",
            "../checkpoints/shallow_impaired",
            "../checkpoints/shallow_impaired_intox",
        ],
        "Single-task MLP",
    )
    outputs = load_checkpoints("../checkpoints/shallow", "Multi-task MLP")

    print()
    cm = outputs["cm3_impaired_first_mean"]
    for row in range(3):
        print(",".join([f"{x:0.2f}" for x in cm[row]]))

    print()
    cm = outputs["cm3_intox_cd_mean"]
    for row in range(3):
        print(",".join([f"{x:0.2f}" for x in cm[row]]))


if __name__ == "__main__":
    main()
