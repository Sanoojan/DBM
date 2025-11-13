import torch.optim as optim


class TorchTrainer:
    def __init__(self, model):
        # Initialize optimizer
        adjusted_lr = model.args.lr * model.args.batch_size
        # TODO(guy.rosman): remove the dependency on batch size / verify
        if model.args.optimizer == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=model.args.weight_decay)
        elif self.model.args.optimizer == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=adjusted_lr, momentum=0.9)
        else:
            raise Exception(f"Unknown optimizer: {model.args.optimizer}")

        # Initialize scheduler
        self.first_batch = True
        if model.args.scheduler is None:
            self.scheduler = None
        elif model.args.scheduler == "linear":
            n_params = len(model.args.scheduler_params)
            if n_params != 1:
                raise Exception(f"Expected 1 parameter for linear scheduler, got {n_params}")
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=model.args.scheduler_params[0],
                end_factor=1.0,
                total_iters=model.args.epochs - 1,
            )
        elif model.args.scheduler == "exponential":
            n_params = len(model.args.scheduler_params)
            if n_params != 1:
                raise Exception(f"Expected 1 parameter for exponential scheduler, got {n_params}")
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=model.args.scheduler_params[0], last_epoch=-1
            )
        else:
            raise Exception(f"Unknown scheduler: {model.args.scheduler}")

    def before_forward(self, batch_idx, train):
        if train and batch_idx == 0 and self.scheduler:
            # Don't step the scheduler until the second epoch
            if self.first_batch:
                self.first_batch = False
            else:
                self.scheduler.step()

    def before_backward(self):
        self.optimizer.zero_grad()

    def after_backward(self):
        self.optimizer.step()
