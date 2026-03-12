"""
Training utilities: seeding, checkpointing, gradient norm.
"""

import os
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic operations (may slow training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set all random seeds to {seed}")


def compute_grad_norm(model: nn.Module) -> float:
    """Compute global gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
):
    """Save training checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    torch.save(state, str(path))

    if is_best:
        best_path = save_dir / "best_model.pt"
        torch.save(state, str(best_path))
        print(f"[Checkpoint] Saved best model to {best_path}")

    return str(path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"[Checkpoint] Loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', '?'):.4f}")

    return checkpoint


class EarlyStopping:
    """Early stopping based on validation accuracy."""

    def __init__(self, patience: int = 12, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_acc = 0.0
        self.should_stop = False

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(
                    f"[EarlyStopping] No improvement for {self.patience} epochs. Stopping."
                )

        return self.should_stop


class TrainingHistory:
    """Stores and saves training metrics history."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "grad_norm": [],
            "epoch_time": [],
        }

    def append(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.history:
                self.history[key].append(val)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            self.history = json.load(f)
        return self.history
