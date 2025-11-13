import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from models.components.aggregate_features import AggregateFeatures
from models.components.torch_discriminator import TorchDisciminator
from models.model_base import ModelBase


class ShallowNet(ModelBase, nn.Module):
    def __init__(self, args, dataloader, device):
        ModelBase.__init__(self, args, dataloader, device)
        nn.Module.__init__(self)

        self.aggregate_features = AggregateFeatures(self)
        self.torch_discriminator = TorchDisciminator(self)

        # Determine model size by processing input for one batch
        batch = next(iter(dataloader))
        X = self.aggregate_features.collect_features(batch)
        n_in = X.shape[-1]

        # Initialize the model
        p = 0.35
        layerlist = []
        layer_channels = [256, 128, 64]
        for n_out in layer_channels:
            layerlist.append(nn.Linear(n_in, n_out))  # n_in input neurons connected to i number of output neurons
            layerlist.append(nn.LeakyReLU())  # Apply activation function - ReLU
            layerlist.append(nn.BatchNorm1d(n_out))  # Apply batch normalization
            layerlist.append(nn.Dropout(p))  # Apply dropout to prevent overfitting
            n_in = n_out
        self.layers = nn.Sequential(*layerlist)
        self.final_layer = nn.Linear(layer_channels[-1], len(args.model_tasks))

        # Send to device
        self.to(self.device)
        self.torch_discriminator.setup_trainer()

    # Returns outputs for the batch (using the previously trained model)
    def forward(self, batch, batch_idx=None):
        outputs = {}
        X = self.aggregate_features.collect_features(batch)
        X = torch.from_numpy(X).float().to(self.device)

        # TODO: Better handle nans in features
        X[torch.isnan(X)] = 0.0

        before_final = self.layers(X)
        outputs["before_final"] = before_final.detach().cpu().numpy()
        outputs["out"] = self.final_layer(before_final)
        return outputs

    # Returns outputs for the batch, including relevant ground truth (if available)
    # Optionally update the model, if requested
    def evaluate_and_update(self, batch, train=False, batch_idx=None):
        return self.torch_discriminator.evaluate_and_update(batch, train, batch_idx)

    # Provides epoch stats for a set of model outputs
    def score(self, outputs):
        return self.torch_discriminator.score(outputs)

    def save(self, path):
        torch.save(self.state_dict(), f"{path}.pt")
