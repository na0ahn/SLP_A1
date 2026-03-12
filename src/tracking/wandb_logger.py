"""
Weights & Biases logger for KWS training.

Falls back to offline mode if W&B authentication is missing.
Always writes local copies of metrics alongside W&B logging.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


class WandBLogger:
    """
    W&B logger wrapper with offline fallback.

    Logs:
      - train/loss, train/accuracy
      - val/loss, val/accuracy
      - learning_rate
      - grad_norm
      - parameter_count
      - epoch_time
      - best_val_accuracy
      - test_accuracy (final)
      - confusion_matrix
      - per_class_accuracy
      - example log-mel images
    """

    def __init__(self, cfg: dict, run_dir: str = "outputs"):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = cfg.get("wandb", {}).get("enabled", True)
        self.run = None
        self._local_history = []

        if self.enabled:
            self._init_wandb(cfg)

    def _init_wandb(self, cfg: dict):
        """Initialize W&B, falling back to offline mode if needed."""
        try:
            import wandb

            wandb_cfg = cfg.get("wandb", {})
            mode = wandb_cfg.get("mode", "offline")

            # Try to set offline mode if requested or if no API key
            if mode == "offline" or not os.environ.get("WANDB_API_KEY"):
                os.environ["WANDB_MODE"] = "offline"
                mode = "offline"

            self.run = wandb.init(
                project=wandb_cfg.get("project", "skku_kws_assignment1"),
                entity=wandb_cfg.get("entity", None),
                config=self._flatten_cfg(cfg),
                tags=wandb_cfg.get("tags", []),
                mode=mode,
                dir=str(self.run_dir),
            )

            print(f"[W&B] Initialized run: {self.run.name} (mode={mode})")
            if mode == "offline":
                print(f"[W&B] Offline mode: logs saved to {self.run_dir}")
                print("[W&B] To sync online: wandb sync <run_dir>/wandb/")

            # Save run URL/path
            run_info = {
                "run_id": self.run.id,
                "run_name": self.run.name,
                "mode": mode,
                "project": wandb_cfg.get("project", "skku_kws_assignment1"),
                "run_dir": str(self.run_dir / "wandb"),
                "sync_command": f"wandb sync {self.run_dir}/wandb/offline-run-*",
            }
            with open(self.run_dir / "wandb_run.txt", "w") as f:
                for k, v in run_info.items():
                    f.write(f"{k}: {v}\n")

        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}")
            print("[W&B] Continuing without W&B tracking.")
            self.enabled = False
            self.run = None

    def _flatten_cfg(self, cfg: dict, prefix: str = "") -> dict:
        """Flatten nested config for W&B."""
        flat = {}
        for k, v in cfg.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_cfg(v, f"{key}/"))
            else:
                flat[key] = v
        return flat

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        grad_norm: float,
        epoch_time: float,
        best_val_acc: float,
        param_count: int,
    ):
        """Log per-epoch metrics."""
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "epoch_time": epoch_time,
            "best_val_accuracy": best_val_acc,
            "parameter_count": param_count,
        }

        self._local_history.append(metrics)
        self._save_local_history()

        if self.enabled and self.run is not None:
            try:
                import wandb
                self.run.log(metrics, step=epoch)
            except Exception as e:
                print(f"[W&B] Log failed: {e}")

    def log_test_results(
        self,
        test_acc: float,
        per_class_acc: dict,
        confusion_matrix: np.ndarray,
        class_names: list,
    ):
        """Log final test results."""
        metrics = {
            "test/accuracy": test_acc,
        }
        for cls, acc in per_class_acc.items():
            metrics[f"test/per_class/{cls}"] = acc

        if self.enabled and self.run is not None:
            try:
                import wandb
                self.run.log(metrics)
                self.run.summary["test_accuracy"] = test_acc
                self.run.summary["best_val_accuracy"] = max(
                    m.get("val/accuracy", 0) for m in self._local_history
                ) if self._local_history else 0
            except Exception as e:
                print(f"[W&B] Test log failed: {e}")

        # Always save locally
        with open(self.run_dir / "test_results.json", "w") as f:
            json.dump({
                "test_accuracy": test_acc,
                "per_class_accuracy": {k: float(v) for k, v in per_class_acc.items()},
            }, f, indent=2)

    def log_images(self, images: dict):
        """Log image artifacts (e.g., log-mel examples)."""
        if self.enabled and self.run is not None:
            try:
                import wandb
                log_dict = {}
                for name, img_array in images.items():
                    log_dict[name] = wandb.Image(img_array)
                self.run.log(log_dict)
            except Exception as e:
                print(f"[W&B] Image log failed: {e}")

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                print("[W&B] Run finished.")
            except Exception as e:
                print(f"[W&B] Finish failed: {e}")

    def _save_local_history(self):
        """Save training history locally as JSON."""
        with open(self.run_dir / "training_history.json", "w") as f:
            json.dump(self._local_history, f, indent=2)

    def get_history(self) -> list:
        return self._local_history
