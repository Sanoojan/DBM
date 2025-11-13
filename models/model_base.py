from abc import ABC, abstractmethod

import numpy as np


class ModelBase(ABC):
    # Takes arguments and returns the model to train
    def __init__(self, args, dataloader, device):
        self.args = args
        self.dataloader = dataloader
        self.device = device
        self.current_fold = None

    # Returns outputs for the batch (using the previously trained model)
    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError()

    # Returns outputs for the batch, including relevant ground truth (if available)
    # Optionally update the model, if requested
    #    Note that training will have no impact on the returned outputs,
    #    as the prior version of the model is used for the forward pass
    @abstractmethod
    def evaluate_and_update(self, batch, train=False, batch_idx=None):
        raise NotImplementedError()

    # Provides epoch stats for a set of model outputs
    @abstractmethod
    def score(self, outputs):
        raise NotImplementedError()

    @abstractmethod
    def save(self, out_path):
        raise NotImplementedError()
