"""
src/training/train_baseline.py
------------------------------
Execution script for training Baseline A (scratch CNN) and Baseline B (frozen ResNet50).

Records:
- Top-1 Accuracy, Macro-F1, Weighted-F1
- Total Parameters & Trainable Parameters
- Total Training Wall-Clock Time
- Saves artifacts to reports/baseline_results.json
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from src.data.class_weights import compute_class_weights
from src.data.dataset import build_dataloader
from src.models.baseline_cnn import build_baseline_cnn
from src.models.transfer import build_model
from src.training.losses import LabelSmoothingCE
from src.training.scheduler import build_scheduler
from src.training.trainer import EarlyStopping, Trainer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seed import set_global_seed


def train_single_baseline(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    cfg,
    device: torch.device,
    logger: Logger,
    epochs: int = 10,
    lr: float = 1e-3,
) -> dict:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler_dict = build_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine_warmup",
        warmup_epochs=2,
        total_epochs=epochs,
    )
    criterion = LabelSmoothingCE(
        num_classes=cfg.data.num_classes,
        smoothing=0.1,
        weight=class_weights if cfg.training.class_weighting else None,
    )
    early_stopping = EarlyStopping(patience=7, mode="max")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler_dict,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        checkpoint_dir=Path(cfg.paths.checkpoints_dir),
        use_amp=cfg.project.mixed_precision,
    )

    t0 = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        model_tag=model_name,
    )
    duration = time.time() - t0

    # Best validation metrics
    best_idx = int(pd.Series(history["val_macro_f1"]).idxmax())
    return {
        "model": model_name,
        "val_acc": history["val_acc"][best_idx],
        "val_macro_f1": history["val_macro_f1"][best_idx],
        "total_params": model.count_parameters()[0],
        "trainable_params": model.count_parameters()[1],
        "train_time_sec": round(duration, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count for baselines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger = Logger("baselines", log_dir=cfg.paths.logs_dir)
    splits_dir = Path(cfg.paths.splits_dir)

    train_csv = splits_dir / "plantvillage_train.csv"
    val_csv = splits_dir / "plantvillage_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        logger.error(f"Splits not found at {splits_dir}. Run `python src/data/splits.py` first.")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    class_weights = compute_class_weights(train_df, num_classes=cfg.data.num_classes)

    train_loader = build_dataloader(
        df=train_df,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.training.stage1.batch_size,
        num_workers=cfg.data.num_workers,
        use_cutmix=False, # standard training for baselines
    )
    val_loader = build_dataloader(
        df=val_df,
        split="val",
        image_size=cfg.data.image_size,
        batch_size=cfg.training.stage1.batch_size,
        num_workers=cfg.data.num_workers,
    )

    results = []

    # 1. Baseline A: Custom CNN Scratch
    logger.info("=== Running Baseline A: Small Custom CNN ===")
    baseline_a = build_baseline_cnn(num_classes=cfg.data.num_classes)
    res_a = train_single_baseline(
        "baseline_cnn",
        baseline_a,
        train_loader,
        val_loader,
        class_weights,
        cfg,
        device,
        logger,
        epochs=args.epochs,
    )
    results.append(res_a)

    # 2. Baseline B: Pretrained ResNet50 Frozen Backbone
    logger.info("=== Running Baseline B: Pretrained ResNet50 (Frozen) ===")
    baseline_b = build_model("resnet50", num_classes=cfg.data.num_classes, pretrained=True, frozen=True)
    res_b = train_single_baseline(
        "resnet50_frozen",
        baseline_b,
        train_loader,
        val_loader,
        class_weights,
        cfg,
        device,
        logger,
        epochs=args.epochs,
    )
    results.append(res_b)

    # Save summary
    out_path = Path(cfg.paths.reports_dir) / "baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print Rich table
    table = Table(title="Baseline Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Val Accuracy", style="magenta")
    table.add_column("Val Macro-F1", style="green")
    table.add_column("Trainable Params", style="yellow")
    table.add_column("Train Time (s)", style="blue")

    for r in results:
        table.add_row(
            r["model"],
            f"{r['val_acc']:.4f}",
            f"{r['val_macro_f1']:.4f}",
            f"{r['trainable_params']:,}",
            f"{r['train_time_sec']}s",
        )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
