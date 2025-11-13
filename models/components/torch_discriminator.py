import numpy as np
import torch
import torch.nn.functional as F

from models.components.discriminator_task import DicriminatorTask
from models.components.torch_trainer import TorchTrainer


class TorchDisciminator:
    def __init__(self, model):
        self.model = model
        self.discriminator_task = DicriminatorTask(model)

        # Determine the output positive weights
        all_labels = self.discriminator_task.convert_ground_truth(
            self.model.dataloader.dataset.scenario_index["cognitive_task"].to_numpy(),
            self.model.dataloader.dataset.scenario_index["intoxicated"].to_numpy(),
        )
        self.pos_weight = (np.sum(all_labels != -1, axis=0) / np.sum(all_labels == 1, axis=0)) - 1
        print("Using pos_weights:", self.pos_weight)
        self.pos_weight = torch.from_numpy(self.pos_weight).float().to(self.model.device)

    def setup_trainer(self):
        self.trainer = TorchTrainer(self.model)

    def evaluate_and_update(self, batch, train=False, batch_idx=None):
        # Setup model
        if train:
            self.model.train()
        else:
            self.model.eval()
        self.trainer.before_forward(batch_idx, train)

        # Forward pass and gather ground truth
        with torch.set_grad_enabled(train):
            outputs = self.model.forward(batch)
        outputs["gt"] = self.discriminator_task.collect_ground_truth(batch)

        # Set mixed ground truth to nan
        gt_tensor = outputs["gt"]
        gt_tensor = torch.from_numpy(outputs["gt"]).float().to(self.model.device)

        # Loss, backprop, and optimizer when training
        if train:
            self.trainer.before_backward()

            task_losses = F.binary_cross_entropy_with_logits(
                outputs["out"], gt_tensor, pos_weight=self.pos_weight, reduction="none"
            )

            # Average the loss using the task weights provided
            loss_total = 0.0
            W_total = 0.0
            for i in range(task_losses.shape[1]):
                is_valid = gt_tensor[:, i] >= 0.0
                total_valid = torch.sum(is_valid)
                if total_valid != 0:
                    task_loss = torch.mean(task_losses[is_valid, i])
                    W = self.model.args.model_task_weights[i] * total_valid
                    loss_total += W * task_loss
                    W_total += W
            loss = loss_total / W_total

            if torch.isnan(loss):
                raise Exception("NaN in loss during training")

            loss.backward()
            outputs["loss"] = loss.detach().item()
            self.trainer.after_backward()

        # Get output probabilities
        outputs["prob"] = F.sigmoid(outputs["out"]).detach().cpu().numpy()
        outputs["out"] = outputs["out"].detach().cpu().numpy()

        return outputs

    # Provides epoch stats for a set of model outputs
    def score(self, outputs):
        return self.discriminator_task.score(outputs)
