import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.components.aggregate_features import AggregateFeatures
from models.components.discriminator_task import DicriminatorTask
from models.model_base import ModelBase


class SklearnModel(ModelBase):
    def __init__(self, args, dataloader, device):
        super().__init__(args, dataloader, device)
        self.task_rf = []
        for _ in range(len(args.model_tasks)):
            if args.model == "RandomForest":
                self.task_rf.append(RandomForestClassifier(class_weight="balanced"))
            elif args.model == "GradientBoosting":
                self.task_rf.append(HistGradientBoostingClassifier(class_weight="balanced"))
            elif args.model == "SVM":
                self.task_rf.append(SVC(class_weight="balanced", probability=True))
            elif args.model == "LogisticRegression":
                self.task_rf.append(LogisticRegression(class_weight="balanced"))
            else:
                raise Exception(f"Unknown model {args.model}")

        # Set up componenets
        self.aggregate_features = AggregateFeatures(self)
        self.discriminator_task = DicriminatorTask(self)

    # Returns outputs for the collected features (using the previously trained model)
    def __inner_forward(self, X):
        outputs = {}

        # Get output probabilities
        probs = []
        for rf in self.task_rf:
            out = rf.predict_proba(X)
            probs.append(out[:, 1])

        outputs["out"] = np.stack(probs, axis=1)
        outputs["prob"] = outputs["out"]

        return outputs

    # Returns outputs for the batch (using the previously trained model)
    def forward(self, batch):
        X = self.aggregate_features.collect_features(batch)

        # Fix nans as needed
        if self.args.model in ["LogisticRegression", "SVM"]:
            X[np.isnan(X)] = 0.0

        return self.__inner_forward(X)

    # Returns outputs for the batch, including relevant ground truth (if available)
    # Optionally update the model, if requested
    def evaluate_and_update(self, batch, train=False, batch_idx=None):
        X = self.aggregate_features.collect_features(batch)

        # Fix nans as needed
        if self.args.model in ["LogisticRegression", "SVM"]:
            X[np.isnan(X)] = 0.0

        if not train:
            outputs = self.__inner_forward(X)
        else:
            outputs = {}
        outputs["gt"] = self.discriminator_task.collect_ground_truth(batch)
        if train:
            for task_idx in range(outputs["gt"].shape[1]):
                use_gt = outputs["gt"][:, task_idx] != -1
                self.task_rf[task_idx].fit(X[use_gt], outputs["gt"][use_gt, task_idx].astype(bool))
        return outputs

    # Provides epoch stats for a set of model outputs
    def score(self, outputs):
        return self.discriminator_task.score(outputs)

    def save(self, path):
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(self.task_rf, file)
