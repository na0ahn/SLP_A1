"""
Evaluation loop for KWS model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from src.engine.metrics import AverageMeter, per_class_accuracy, confusion_matrix_np
from src.engine.losses import LabelSmoothingCrossEntropy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> dict:
    """
    Evaluate model on a data loader.

    Returns:
        dict with keys: loss, accuracy, per_class_acc, confusion_matrix,
                        all_preds, all_targets
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=-1)

            loss_meter.update(loss.item(), inputs.size(0))
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    # Overall accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    total = len(all_targets)
    acc = correct / total

    # Per-class accuracy
    pca = per_class_accuracy(all_preds, all_targets, class_names)

    # Confusion matrix
    cm = confusion_matrix_np(all_preds, all_targets, len(class_names))

    return {
        "loss": loss_meter.avg,
        "accuracy": acc,
        "per_class_acc": pca,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "n_correct": correct,
        "n_total": total,
    }
