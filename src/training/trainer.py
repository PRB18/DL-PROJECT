"""
src/training/trainer.py
-----------------------
Training and validation orchestrator supporting:
- Automatic Mixed Precision (AMP with torch.amp.autocast)
- CutMix loss computation
- Gradient clipping
- Early stopping monitored on validation macro-F1
- Best checkpoint saving (weights + metadata + optimizer state)
- Per-epoch wall-clock timing and TensorBoard logging
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import compute_cutmix_loss
from src.training.scheduler import get_lr


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def compute_epoch_metrics(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
) -> Dict[str, float]:
    """Calculate core classification metrics."""
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


class Trainer:
    """
    Production-grade model trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        criterion: nn.Module,
        device: torch.device,
        cfg: Any,
        logger: Any,
        checkpoint_dir: Path,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and (device.type == "cuda")

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_macro_f1 = 0.0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                if batch.get("cutmix", False):
                    loss = compute_cutmix_loss(
                        logits=logits,
                        label_a=labels,
                        label_b=batch["label_b"].to(self.device, non_blocking=True),
                        lam=batch["lam"],
                        criterion=self.criterion,
                    )
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(dataloader.dataset)
        metrics = compute_epoch_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = epoch_loss
        metrics["lr"] = get_lr(self.optimizer)
        return metrics

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())

        epoch_loss = total_loss / len(dataloader.dataset)
        metrics = compute_epoch_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = epoch_loss
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        model_tag: str = "model",
    ) -> Dict[str, list]:
        history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_macro_f1": [], "val_macro_f1": [],
            "lr": [],
        }

        self.logger.info(f"Starting training run: {model_tag} for {epochs} epochs on {self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader, epoch)
            val_m = self.val_epoch(val_loader, epoch)
            elapsed = time.time() - t0

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, dict):
                    self.scheduler["scheduler"].step()
                else:
                    self.scheduler.step()

            # Record history
            history["train_loss"].append(train_m["loss"])
            history["val_loss"].append(val_m["loss"])
            history["train_acc"].append(train_m["accuracy"])
            history["val_acc"].append(val_m["accuracy"])
            history["train_macro_f1"].append(train_m["macro_f1"])
            history["val_macro_f1"].append(val_m["macro_f1"])
            history["lr"].append(train_m["lr"])

            # Log to console & TensorBoard
            log_str = (
                f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} F1: {train_m['macro_f1']:.4f} | "
                f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} F1: {val_m['macro_f1']:.4f} | "
                f"LR: {train_m['lr']:.2e}"
            )
            self.logger.info(log_str)

            self.logger.scalar(f"{model_tag}/train_loss", train_m["loss"], epoch)
            self.logger.scalar(f"{model_tag}/val_loss", val_m["loss"], epoch)
            self.logger.scalar(f"{model_tag}/val_macro_f1", val_m["macro_f1"], epoch)

            # Checkpoint best
            if val_m["macro_f1"] > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_m["macro_f1"]
                self.save_checkpoint(epoch, val_m, filename=f"best_{model_tag}.pth")
                self.logger.info(f"[*] New best validation macro-F1: {self.best_val_macro_f1:.4f}. Saved checkpoint.")

            # Early stopping check
            if early_stopping and early_stopping(val_m["macro_f1"]):
                self.logger.info(f"[!] Early stopping triggered at epoch {epoch}.")
                break

        # Save final checkpoint
        self.save_checkpoint(epoch, val_m, filename=f"final_{model_tag}.pth")
        return history

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str) -> Path:
        save_path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, save_path)
        return save_path

    def load_checkpoint(self, path: Path) -> Tuple[int, Dict[str, float]]:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
