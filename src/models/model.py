"""
Model factory and parameter validation.

Usage:
    from src.models.model import build_model, validate_param_count
"""

import os
import torch
import torch.nn as nn


MAX_PARAMS = 2_500_000  # Hard constraint from assignment


def build_model(cfg: dict) -> nn.Module:
    """
    Build model from config.

    Performs hard runtime assertion that parameter count <= 2.5M.
    """
    model_name = cfg.get("model", {}).get("name", "dscnn").lower()

    if model_name == "dscnn":
        from src.models.dscnn import build_dscnn
        model = build_dscnn(cfg)
    elif model_name == "bcresnet":
        from src.models.bcresnet import build_bcresnet
        model = build_bcresnet(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'dscnn' or 'bcresnet'.")

    # Hard runtime parameter count assertion (assignment requirement)
    validate_param_count(model)

    return model


def validate_param_count(model: nn.Module) -> int:
    """
    Count trainable parameters and assert <= 2.5M.

    Raises:
        AssertionError if parameter count exceeds limit.

    Returns:
        int: parameter count
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Model] Trainable parameters: {n_params:,}")
    print(f"[Model] Parameter budget: {MAX_PARAMS:,}")
    print(f"[Model] Budget utilization: {100 * n_params / MAX_PARAMS:.1f}%")

    assert n_params <= MAX_PARAMS, (
        f"HARD CONSTRAINT VIOLATED: Model has {n_params:,} parameters "
        f"which exceeds the 2.5M limit ({MAX_PARAMS:,})! "
        f"Reduce model size."
    )

    print(f"[Model] Parameter count check PASSED ({n_params:,} <= {MAX_PARAMS:,})\n")
    return n_params


def save_model_summary(model: nn.Module, cfg: dict, save_path: str):
    """Save model architecture summary to text file."""
    import json
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    lines = [
        "=" * 70,
        "KWS Model Summary",
        "=" * 70,
        f"Model type       : {cfg.get('model', {}).get('name', 'dscnn').upper()}",
        f"Trainable params : {n_params:,}",
        f"Total params     : {n_total:,}",
        f"Max allowed      : {MAX_PARAMS:,}",
        f"Budget used      : {100 * n_params / MAX_PARAMS:.1f}%",
        "",
        "Architecture:",
        "-" * 70,
        str(model),
        "",
        "=" * 70,
        "Feature spec:",
        f"  Input shape    : (1, 80, 49)  [channels, n_mels, n_frames]",
        f"  sample_rate    : 16000 Hz",
        f"  n_fft          : 1024",
        f"  win_length     : 640 samples (40 ms)",
        f"  hop_length     : 320 samples (20 ms overlap)",
        f"  n_mels         : 80",
        f"  f_min          : 20 Hz",
        f"  f_max          : 8000 Hz",
        f"  log transform  : log(mel + 1e-6)",
        f"  normalization  : utterance-wise MVN",
        "=" * 70,
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Model] Summary saved to {save_path}")
    return n_params
