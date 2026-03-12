"""
Visualization utilities for log-mel features and augmentations.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch


def plot_waveform_and_logmel(
    waveform: torch.Tensor,
    log_mel: torch.Tensor,
    title: str = "",
    sample_rate: int = 16000,
    ax_wave=None,
    ax_mel=None,
):
    """Plot waveform and log-mel spectrogram side by side."""
    if ax_wave is None or ax_mel is None:
        fig, (ax_wave, ax_mel) = plt.subplots(1, 2, figsize=(10, 3))
        standalone = True
    else:
        standalone = False

    # Waveform
    wv = waveform.squeeze().numpy() if isinstance(waveform, torch.Tensor) else waveform
    t = np.linspace(0, len(wv) / sample_rate, len(wv))
    ax_wave.plot(t, wv, linewidth=0.5, color="steelblue")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"Waveform: {title}")
    ax_wave.set_xlim([0, len(wv) / sample_rate])

    # Log-mel
    lm = log_mel.squeeze().numpy() if isinstance(log_mel, torch.Tensor) else log_mel
    if lm.ndim == 3:
        lm = lm[0]
    img = ax_mel.imshow(
        lm, aspect="auto", origin="lower",
        cmap="viridis", interpolation="nearest"
    )
    ax_mel.set_xlabel("Time frames")
    ax_mel.set_ylabel("Mel bins")
    ax_mel.set_title(f"Log-Mel: {title}")
    plt.colorbar(img, ax=ax_mel, label="Log amplitude")

    if standalone:
        plt.tight_layout()
        return plt.gcf()


def save_feature_examples(
    dataset,
    class_names: list,
    save_path: str,
    n_per_class: int = 1,
    seed: int = 42,
):
    """
    Save example waveforms and log-mel spectrograms for each class.

    Args:
        dataset: SpeechCommandsDataset
        class_names: list of class names
        save_path: path to save the figure
        n_per_class: number of examples per class
    """
    import random
    rng = random.Random(seed)

    # Group items by class
    from collections import defaultdict
    class_items = defaultdict(list)
    for i, item in enumerate(dataset.items):
        class_items[item["class_idx"]].append(i)

    # Select one example per class
    selected = {}
    for cls_idx, name in enumerate(class_names):
        if cls_idx in class_items and class_items[cls_idx]:
            idx = rng.choice(class_items[cls_idx])
            selected[name] = idx

    n_classes = len(selected)
    fig, axes = plt.subplots(n_classes, 2, figsize=(12, 3 * n_classes))

    for row, (name, idx) in enumerate(sorted(selected.items())):
        log_mel, label = dataset[idx]
        # Get waveform too
        from src.data.dataset import load_and_preprocess
        item = dataset.items[idx]
        if item.get("is_noise", False):
            waveform = dataset._load_noise_chunk(item)
        else:
            waveform = load_and_preprocess(item["path"])

        plot_waveform_and_logmel(
            waveform, log_mel,
            title=name,
            ax_wave=axes[row, 0] if n_classes > 1 else axes[0],
            ax_mel=axes[row, 1] if n_classes > 1 else axes[1],
        )

    plt.suptitle("Log-Mel Feature Examples by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved feature examples to {save_path}")


def save_augmentation_examples(
    dataset,
    class_names: list,
    save_path: str,
    n_examples: int = 4,
    seed: int = 42,
):
    """
    Save original vs augmented log-mel examples.

    Shows the effect of RandomTimeShift + SpecAugment.
    """
    import random
    from src.data.transforms import RandomTimeShift, SpecAugment
    from src.data.dataset import load_and_preprocess

    rng = random.Random(seed)
    random.seed(seed)

    # Pick random samples (non-silence)
    target_items = [
        (i, item) for i, item in enumerate(dataset.items)
        if item.get("label", "") not in ("silence", "unknown") and not item.get("is_noise", False)
    ]

    rng.shuffle(target_items)
    selected = target_items[:n_examples]

    time_shift = RandomTimeShift(max_shift_ms=100)
    spec_aug = SpecAugment(time_masks=2, time_width=8, freq_masks=2, freq_width=8)

    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 3 * n_examples))

    for row, (idx, item) in enumerate(selected):
        waveform = load_and_preprocess(item["path"])
        log_mel_orig = dataset.extractor(waveform)

        # Apply augmentations
        waveform_aug = time_shift(waveform)
        log_mel_aug = dataset.extractor(waveform_aug)
        log_mel_specaug = spec_aug(log_mel_aug.clone())

        lm_o = log_mel_orig.numpy()
        lm_a = log_mel_aug.numpy()
        lm_s = log_mel_specaug.numpy()

        vmin = min(lm_o.min(), lm_a.min(), lm_s.min())
        vmax = max(lm_o.max(), lm_a.max(), lm_s.max())

        label = item["label"]

        ax = axes[row, 0] if n_examples > 1 else axes[0]
        ax.imshow(lm_o, aspect="auto", origin="lower", cmap="viridis",
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"Original ({label})")
        ax.set_ylabel("Mel bins")

        ax = axes[row, 1] if n_examples > 1 else axes[1]
        ax.imshow(lm_a, aspect="auto", origin="lower", cmap="viridis",
                  vmin=vmin, vmax=vmax)
        ax.set_title("After TimeShift")

        ax = axes[row, 2] if n_examples > 1 else axes[2]
        ax.imshow(lm_s, aspect="auto", origin="lower", cmap="viridis",
                  vmin=vmin, vmax=vmax)
        ax.set_title("After SpecAugment")

    plt.suptitle("Augmentation Examples: Original → TimeShift → SpecAugment",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved augmentation examples to {save_path}")
