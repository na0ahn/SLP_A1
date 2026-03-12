"""
Main training loop for KWS model.

Implements:
  - AdamW optimizer with cosine decay + warmup
  - Gradient clipping
  - Mixed precision (optional)
  - Early stopping
  - W&B logging
  - Checkpoint saving (best by val accuracy)

Regularization:
  1. Dropout (in model)
  2. Weight decay (in AdamW)
  3. Label smoothing (in loss function)

Augmentation:
  1. RandomTimeShift (waveform)
  2. SpecAugment (spectrogram)
"""

import os
import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.losses import build_criterion
from src.engine.scheduler import build_optimizer, build_scheduler
from src.engine.metrics import AverageMeter, accuracy
from src.engine.evaluate import evaluate
from src.engine.utils import (
    compute_grad_norm, save_checkpoint, EarlyStopping, TrainingHistory
)
from src.tracking.wandb_logger import WandBLogger


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    log_interval: int = 50,
    use_amp: bool = False,
    scaler=None,
) -> dict:
    """
    Train for one epoch.

    Returns:
        dict with train_loss, train_acc, grad_norm
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_norms = []

    total_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            grad_norm = compute_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_acc = accuracy(logits.detach(), targets)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(batch_acc, inputs.size(0))
        grad_norms.append(grad_norm)

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  [{batch_idx + 1}/{total_batches}] "
                f"loss={loss_meter.avg:.4f}  acc={acc_meter.avg:.4f}  "
                f"grad_norm={grad_norm:.3f}"
            )

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

    return {
        "train_loss": loss_meter.avg,
        "train_acc": acc_meter.avg,
        "grad_norm": avg_grad_norm,
    }


def train(cfg: dict):
    """
    Full training pipeline.

    Args:
        cfg: configuration dict (loaded from YAML)

    Returns:
        dict with final results
    """
    from src.engine.utils import set_seed
    from src.data.dataset import make_dataloaders
    from src.models.model import build_model, save_model_summary

    # ── Setup ─────────────────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    ckpt_dir = Path(train_cfg.get("save_dir", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[Train] Loading dataset...")
    train_loader, valid_loader, test_loader, class_names = make_dataloaders(cfg, seed=seed)

    print(f"[Train] Train batches: {len(train_loader)}")
    print(f"[Train] Valid batches: {len(valid_loader)}")
    print(f"[Train] Test  batches: {len(test_loader)}")

    # Save data summary
    from src.data.dataset import save_split_summary
    save_split_summary(
        train_loader.dataset.items,
        valid_loader.dataset.items,
        test_loader.dataset.items,
        out_dir=str(out_dir / "summaries"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[Train] Building model...")
    model = build_model(cfg)
    model = model.to(device)

    n_params = model.count_parameters()

    # Save model summary
    save_model_summary(model, cfg, str(out_dir / "model_summary.txt"))

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Mixed precision (disabled on CPU)
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── W&B Logger ────────────────────────────────────────────────────────────
    logger = WandBLogger(cfg, run_dir=str(out_dir))

    # ── Training state ────────────────────────────────────────────────────────
    epochs = train_cfg.get("epochs", 40)
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    log_interval = train_cfg.get("log_interval", 50)
    patience = train_cfg.get("early_stopping_patience", 12)

    history = TrainingHistory()
    early_stop = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n[Train] Starting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip,
            log_interval=log_interval,
            use_amp=use_amp,
            scaler=scaler,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        # ── Update scheduler ──────────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Track metrics ─────────────────────────────────────────────────────
        t_epoch = time.time() - t0
        is_best = val_metrics["accuracy"] > best_val_acc

        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch

        history.append(
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_acc"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            lr=current_lr,
            grad_norm=train_metrics["grad_norm"],
            epoch_time=t_epoch,
        )

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            train_acc=train_metrics["train_acc"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            lr=current_lr,
            grad_norm=train_metrics["grad_norm"],
            epoch_time=t_epoch,
            best_val_acc=best_val_acc,
            param_count=n_params,
        )

        print(
            f"  train_loss={train_metrics['train_loss']:.4f}  "
            f"train_acc={train_metrics['train_acc']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"lr={current_lr:.6f}  "
            f"time={t_epoch:.1f}s  "
            f"{'★ best' if is_best else ''}"
        )

        # ── Save checkpoint ───────────────────────────────────────────────────
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "val_acc": val_metrics["accuracy"],
            "train_acc": train_metrics["train_acc"],
            "train_loss": train_metrics["train_loss"],
            "cfg": cfg,
            "class_names": class_names,
            "n_params": n_params,
        }

        save_checkpoint(
            checkpoint,
            save_dir=str(ckpt_dir),
            filename=f"checkpoint_epoch{epoch:03d}.pt",
            is_best=is_best,
        )

        # Also save latest
        save_checkpoint(
            checkpoint,
            save_dir=str(ckpt_dir),
            filename="checkpoint_latest.pt",
            is_best=False,
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stop(val_metrics["accuracy"]):
            print(f"[Train] Early stopping at epoch {epoch}.")
            break

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = str(out_dir / "logs" / "training_history.json")
    history.save(history_path)
    print(f"[Train] History saved to {history_path}")

    # ── Final test evaluation (using best checkpoint) ─────────────────────────
    print(f"\n[Train] Loading best model from epoch {best_epoch}...")
    best_ckpt = torch.load(str(ckpt_dir / "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    print("[Train] Running final test evaluation...")
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

    test_acc = test_metrics["accuracy"]
    print(f"\n[Train] ============================================")
    print(f"[Train] FINAL RESULTS:")
    print(f"[Train]   Best validation accuracy: {best_val_acc:.4f} ({100*best_val_acc:.2f}%)")
    print(f"[Train]   Test accuracy:             {test_acc:.4f} ({100*test_acc:.2f}%)")
    print(f"[Train] ============================================\n")

    # Log test results
    logger.log_test_results(
        test_acc=test_acc,
        per_class_acc=test_metrics["per_class_acc"],
        confusion_matrix=test_metrics["confusion_matrix"],
        class_names=class_names,
    )

    # ── Save final results JSON ───────────────────────────────────────────────
    final_results = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "n_params": n_params,
        "per_class_test_acc": {k: float(v) for k, v in test_metrics["per_class_acc"].items()},
        "class_names": class_names,
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }

    results_path = str(out_dir / "summaries" / "final_results.json")
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"[Train] Results saved to {results_path}")

    logger.finish()

    return {
        "final_results": final_results,
        "history": history.history,
        "test_metrics": test_metrics,
        "class_names": class_names,
        "best_ckpt_path": str(ckpt_dir / "best_model.pt"),
    }
