"""
Learning rate scheduler with linear warmup + cosine decay.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_epochs` then cosine annealing to min_lr.

    Args:
        optimizer:      PyTorch optimizer
        warmup_epochs:  number of warmup epochs
        total_epochs:   total training epochs
        min_lr:         minimum learning rate (default 1e-6)
        last_epoch:     last epoch index (default -1)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale to [min_lr/base_lr, 1.0]
            scale = self.min_lr / self.base_lrs[0] + (1.0 - self.min_lr / self.base_lrs[0]) * scale

        return [base_lr * scale for base_lr in self.base_lrs]


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer from config."""
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # Regularization technique #2
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def build_scheduler(optimizer, cfg: dict) -> WarmupCosineScheduler:
    """Build warmup+cosine scheduler from config."""
    train_cfg = cfg.get("training", {})
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg.get("epochs", 40),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
    )
