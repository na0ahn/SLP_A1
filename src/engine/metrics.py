"""
Evaluation metrics for KWS.
"""

import torch
import numpy as np
from typing import List, Tuple


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def per_class_accuracy(
    all_preds: List[int],
    all_targets: List[int],
    class_names: List[str],
) -> dict:
    """Per-class accuracy."""
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    result = {}
    for i, name in enumerate(class_names):
        mask = targets == i
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = (preds[mask] == i).mean()

    return result


def confusion_matrix_np(
    all_preds: List[int],
    all_targets: List[int],
    n_classes: int,
) -> np.ndarray:
    """Build confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        cm[t][p] += 1
    return cm


class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
