#!/usr/bin/env python3
"""
Generate data/feature visualization figures.

Run this after the dataset is prepared but before (or during) training.
Generates:
  - report_assets/feature_examples.png
  - report_assets/augmentation_examples.png
  - report_assets/class_distribution.png

Usage:
    python generate_data_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    import yaml
    import torch
    from src.data.dataset import build_splits, SpeechCommandsDataset, save_split_summary, ALL_LABELS
    from src.features.logmel import LogMelExtractor
    from src.data.transforms import RandomTimeShift, SpecAugment
    from src.features.visualize import save_feature_examples, save_augmentation_examples
    from src.report.export_assets import plot_class_distribution

    print("Generating data and feature visualization figures...")

    # Load config
    with open("configs/final.yaml") as f:
        cfg = yaml.safe_load(f)

    # Build splits
    data_root = cfg["data"]["root"]
    print(f"Loading dataset from {data_root}...")
    train_items, valid_items, test_items = build_splits(data_root, seed=42)

    # Save data summary
    summary = save_split_summary(train_items, valid_items, test_items, "outputs/summaries")

    # Plot class distribution
    plot_class_distribution(summary, "report_assets/class_distribution.png")
    import shutil
    shutil.copy("report_assets/class_distribution.png", "outputs/class_distribution.png")

    # Feature examples
    feat_cfg = cfg["feature"]
    extractor = LogMelExtractor(
        sample_rate=cfg["data"]["sample_rate"],
        n_fft=feat_cfg["n_fft"],
        win_length=feat_cfg["win_length"],
        hop_length=feat_cfg["hop_length"],
        n_mels=feat_cfg["n_mels"],
        f_min=feat_cfg["f_min"],
        f_max=feat_cfg["f_max"],
        log_eps=feat_cfg["log_eps"],
        center=feat_cfg["center"],
        feature_norm=feat_cfg["feature_norm"],
    )

    ds = SpeechCommandsDataset(
        train_items, extractor,
        augment_waveform=[RandomTimeShift()],
        augment_spec=[SpecAugment()],
        training=False,  # No augmentation for feature examples
    )

    save_feature_examples(
        ds, ALL_LABELS,
        save_path="report_assets/feature_examples.png",
        n_per_class=1,
    )

    # Augmentation examples
    save_augmentation_examples(
        ds, ALL_LABELS,
        save_path="report_assets/augmentation_examples.png",
        n_examples=4,
    )

    print("\nData figures generated:")
    for f in Path("report_assets").glob("*.png"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
