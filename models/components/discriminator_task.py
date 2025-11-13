import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


class DicriminatorTask:
    def __init__(self, model):
        self.model = model

    # Takes a numpy array of cognitive_task and intoxicated and returns the ground truth used by the model
    def convert_ground_truth(self, cognitive_task, intoxicated):
        ground_truth = []
        cd = cognitive_task != "no_task"
        intox = intoxicated
        impaired = np.logical_or(cd, intox)
        for model_task in self.model.args.model_tasks:
            if model_task == "cd":
                ground_truth.append(cd.astype(int))
            elif model_task == "nback":
                ground_truth.append((cognitive_task == "nback_task").astype(int))
            elif model_task == "statement":
                ground_truth.append((cognitive_task == "statement_task").astype(int))
            elif model_task == "intoxicated":
                ground_truth.append(intox.astype(int))
            elif model_task == "impaired":
                ground_truth.append(impaired.astype(int))
            elif model_task == "impaired_intoxicated":
                impaired_intoxicated = intox.copy().astype(int)
                impaired_intoxicated[np.logical_not(impaired)] = -1
                ground_truth.append(impaired_intoxicated)
            elif model_task == "sober_cd":
                sober_cd = cd.copy().astype(int)
                sober_cd[intox] = -1
                ground_truth.append(sober_cd)
            else:
                raise Exception(f"Unknown model output task {model_task}")
        ground_truth = np.stack(ground_truth, axis=-1)
        return ground_truth

    # Gathers the ground truth used for the model
    def collect_ground_truth(self, batch):
        return self.convert_ground_truth(batch.scenario.cognitive_task, batch.scenario.intoxicated)

    # Provides epoch stats for a set of model outputs
    def score(self, outputs):
        epoch_stats = {}
        task_stats = {}
        task_weights = {}
        for task_name, model_weight in zip(self.model.args.model_tasks, self.model.args.model_task_weights):
            task_stats[task_name] = {}
            task_weights[task_name] = model_weight

        # Save aggregate loss (averaged over all batches)
        # TODO: Consider weighting by batch size to account for final batch
        if "loss" in outputs[0]:
            epoch_stats["loss"] = np.mean([x["loss"] for x in outputs])

        # Metrics using probability prediction
        if "prob" in outputs[0]:
            all_gt = np.concatenate([x["gt"] for x in outputs], axis=0)
            all_prob = np.concatenate([x["prob"] for x in outputs], axis=0)

            for task_on, task_name in enumerate(self.model.args.model_tasks):
                task_prob = all_prob[..., task_on]
                task_gt = all_gt[..., task_on]
                use_gt = task_gt != -1
                task_prob = task_prob[use_gt]
                task_pred = task_prob >= 0.5
                task_gt = task_gt[use_gt].astype(bool)

                tn, fp, fn, tp = confusion_matrix(task_gt, task_pred, labels=[False, True]).ravel()

                sensitivity = tp / (tp + fn)
                specificity = tn / (fp + tn)

                uar = (specificity + sensitivity) / 2.0

                accuracy = (tp + tn) / (tp + tn + fp + fn)

                task_stats[task_name]["sensitivity"] = -sensitivity
                task_stats[task_name]["specificity"] = -specificity
                task_stats[task_name]["accuracy"] = -accuracy
                task_stats[task_name]["uar"] = -uar

                task_stats[task_name]["f1"] = -f1_score(task_gt, task_pred, zero_division=np.nan)

                try:
                    task_stats[task_name]["auc"] = -roc_auc_score(task_gt, task_prob)
                except:
                    task_stats[task_name]["auc"] = np.nan

        # Calculate joint stats
        stat_names = set()
        for task_name, single_task_stats in task_stats.items():
            stat_names = stat_names.union(single_task_stats.keys())
        stat_names = list(stat_names)
        for stat_name in stat_names:
            stat_total = 0.0
            weight_total = 0.0
            for task_name, single_task_stats in task_stats.items():
                if stat_name in single_task_stats:
                    W = task_weights[task_name]
                    stat_total += W * single_task_stats[stat_name]
                    weight_total += W
            epoch_stats[f"joint_{stat_name}"] = stat_total / weight_total if weight_total != 0.0 else np.nan

        # Unpack tasks into epoch stats
        for task_name, single_task_stats in task_stats.items():
            for stat_name, stat_value in single_task_stats.items():
                epoch_stats[f"{task_name}_{stat_name}"] = stat_value

        return epoch_stats
