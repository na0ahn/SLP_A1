"""
Loss functions for KWS training.

Uses cross-entropy with label smoothing as primary loss.
Label smoothing is a regularization technique (one of the required >=2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Regularization technique #3 (alongside dropout and weight decay).

    Args:
        smoothing: label smoothing factor (0.0 = standard CE, default 0.05)
        num_classes: number of output classes
    """

    def __init__(self, smoothing: float = 0.05, num_classes: int = 12):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) unnormalized predictions
            targets: (B,)  ground truth class indices

        Returns:
            scalar loss
        """
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, targets)

        log_probs = F.log_softmax(logits, dim=-1)

        # Hard label component
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Smooth component: uniform over all classes
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def build_criterion(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    train_cfg = cfg.get("training", {})
    smoothing = train_cfg.get("label_smoothing", 0.05)
    n_classes = cfg.get("model", {}).get("n_classes", 12)

    return LabelSmoothingCrossEntropy(smoothing=smoothing, num_classes=n_classes)
