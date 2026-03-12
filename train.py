#!/usr/bin/env python3
"""
KWS Training Entry Point.

Usage:
    python train.py                           # use configs/final.yaml
    python train.py --config configs/default.yaml
    python train.py --config configs/final.yaml --epochs 40
    python train.py --sanity                  # quick sanity run (3 epochs)

Assignment: Keyword Spotting on Google Speech Commands v2
  - 12-class setup: 10 target words + silence + unknown
  - Log-Mel features: 80 mel bins, 40ms window, 20ms hop
  - Model: DS-CNN with residual connections (~750K params)
  - Augmentation: RandomTimeShift + SpecAugment
  - Regularization: Dropout + Weight Decay + Label Smoothing
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: dict, overrides: dict) -> dict:
    """Apply command-line overrides to config."""
    for key, val in overrides.items():
        # Handle nested keys like "training.epochs"
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="KWS Training on GSCv2")
    parser.add_argument("--config", default="configs/final.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--model", type=str, default=None,
                        choices=["dscnn", "bcresnet"],
                        help="Override model type")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="Override W&B mode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--sanity", action="store_true",
                        help="Quick sanity run: 3 epochs, small subset")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation on best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval_only mode")
    return parser.parse_args()


def run_sanity_check(cfg: dict):
    """Quick sanity check: load data, run 2 epochs."""
    print("\n" + "=" * 60)
    print("SANITY CHECK MODE: 2 epochs")
    print("=" * 60 + "\n")

    cfg["training"]["epochs"] = 2
    cfg["training"]["early_stopping_patience"] = 99
    cfg["training"]["log_interval"] = 10
    cfg["data"]["num_workers"] = 0  # Simpler for sanity
    cfg["wandb"]["mode"] = "disabled"

    from src.engine.train import train
    results = train(cfg)

    print("\n[Sanity] Sanity check completed successfully!")
    print(f"[Sanity] Val accuracy after 2 epochs: {results['final_results']['best_val_accuracy']:.4f}")
    return results


def run_training(cfg: dict):
    """Run full training pipeline."""
    print("\n" + "=" * 60)
    print("FULL TRAINING")
    print(f"Epochs: {cfg['training']['epochs']}")
    print(f"Model:  {cfg['model']['name']}")
    print("=" * 60 + "\n")

    from src.engine.train import train
    results = train(cfg)
    return results


def run_eval_only(cfg: dict, checkpoint_path: str):
    """Run evaluation on a saved checkpoint."""
    import torch
    from src.models.model import build_model
    from src.data.dataset import make_dataloaders
    from src.engine.evaluate import evaluate
    from src.engine.losses import build_criterion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Use config from checkpoint if available
    saved_cfg = ckpt.get("cfg", cfg)
    model = build_model(saved_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    _, _, test_loader, class_names = make_dataloaders(saved_cfg)
    criterion = build_criterion(saved_cfg)

    test_metrics = evaluate(model, test_loader, criterion, device, class_names)
    print(f"[Eval] Test accuracy: {test_metrics['accuracy']:.4f} ({100*test_metrics['accuracy']:.2f}%)")
    return test_metrics


def main():
    args = parse_args()

    # Load config
    if not Path(args.config).exists():
        print(f"Config not found: {args.config}, using configs/default.yaml")
        args.config = "configs/default.yaml"

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.wandb_mode is not None:
        cfg["wandb"]["mode"] = args.wandb_mode
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    # Print config summary
    print(f"\n[Config] Loaded from: {args.config}")
    print(f"[Config] Model:  {cfg['model']['name']}")
    print(f"[Config] Epochs: {cfg['training']['epochs']}")
    print(f"[Config] Batch:  {cfg['training']['batch_size']}")
    print(f"[Config] LR:     {cfg['training']['lr']}")
    print(f"[Config] W&B:    {cfg['wandb']['mode']}")

    # Run
    if args.sanity:
        run_sanity_check(cfg)
    elif args.eval_only:
        ckpt_path = args.checkpoint or "outputs/checkpoints/best_model.pt"
        run_eval_only(cfg, ckpt_path)
    else:
        results = run_training(cfg)

        # Generate report assets
        print("\n[Main] Generating report assets...")
        try:
            from src.report.export_assets import export_all_assets
            export_all_assets()
            print("[Main] Report assets generated successfully.")
        except Exception as e:
            print(f"[Main] Report asset generation failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
