"""
src/training/train_transfer.py
------------------------------
Two-Stage Transfer Learning for:
- EfficientNetV2-S
- ConvNeXt-Tiny
- MobileNetV3-Large

Stage 1: Train classification head with frozen backbone.
Stage 2: Unfreeze backbone and fine-tune with discriminative learning rates.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from src.data.class_weights import compute_class_weights
from src.data.dataset import build_dataloader
from src.models.transfer import build_model
from src.training.losses import LabelSmoothingCE
from src.training.scheduler import build_scheduler
from src.training.trainer import EarlyStopping, Trainer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seed import set_global_seed


def train_two_stage(
    model_name: str,
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_weights: torch.Tensor,
    device: torch.device,
    logger: Logger,
    image_size: int = 224,
) -> Dict:
    logger.info(f"\n==========================================")
    logger.info(f"   Transfer Learning: {model_name.upper()}")
    logger.info(f"==========================================")

    # Dataloaders
    train_loader = build_dataloader(
        df=train_df,
        split="train",
        image_size=image_size,
        batch_size=cfg.training.stage1.batch_size,
        num_workers=cfg.data.num_workers,
        use_cutmix=True,
        cutmix_prob=cfg.augmentation.train.cutmix_prob,
        background_rand=cfg.augmentation.train.background_randomize,
    )
    val_loader = build_dataloader(
        df=val_df,
        split="val",
        image_size=image_size,
        batch_size=cfg.training.stage1.batch_size,
        num_workers=cfg.data.num_workers,
    )

    criterion = LabelSmoothingCE(
        num_classes=cfg.data.num_classes,
        smoothing=cfg.training.stage1.label_smoothing,
        weight=class_weights if cfg.training.class_weighting else None,
    )

    model = build_model(model_name, num_classes=cfg.data.num_classes, pretrained=True, frozen=True)
    t0 = time.time()

    # --- STAGE 1: Head Only ---
    logger.info(f"\n--- [Stage 1] Training classification head (frozen backbone) ---")
    opt_stage1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.stage1.lr,
        weight_decay=cfg.training.stage1.weight_decay,
    )
    sched_stage1 = build_scheduler(
        optimizer=opt_stage1,
        scheduler_type="cosine_warmup",
        warmup_epochs=cfg.training.stage1.warmup_epochs,
        total_epochs=cfg.training.stage1.epochs,
    )
    trainer1 = Trainer(
        model=model,
        optimizer=opt_stage1,
        scheduler=sched_stage1,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        checkpoint_dir=Path(cfg.paths.checkpoints_dir),
        use_amp=cfg.project.mixed_precision,
    )
    trainer1.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.stage1.epochs,
        early_stopping=EarlyStopping(patience=5, mode="max"),
        model_tag=f"{model_name}_stage1",
    )

    # --- STAGE 2: Unfreeze with Discriminative LRs ---
    logger.info(f"\n--- [Stage 2] Fine-tuning all layers (discriminative LRs) ---")
    model.unfreeze_backbone()

    param_groups = model.get_param_groups(
        backbone_lr_factor=cfg.training.stage2.backbone_lr_factor,
        head_lr=cfg.training.stage2.lr,
        weight_decay=cfg.training.stage2.weight_decay,
    )
    opt_stage2 = torch.optim.AdamW(param_groups)
    sched_stage2 = build_scheduler(
        optimizer=opt_stage2,
        scheduler_type="cosine_warmup",
        warmup_epochs=cfg.training.stage2.warmup_epochs,
        total_epochs=cfg.training.stage2.epochs,
    )

    trainer2 = Trainer(
        model=model,
        optimizer=opt_stage2,
        scheduler=sched_stage2,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        checkpoint_dir=Path(cfg.paths.checkpoints_dir),
        use_amp=cfg.project.mixed_precision,
    )
    hist2 = trainer2.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.stage2.epochs,
        early_stopping=EarlyStopping(patience=cfg.training.early_stopping.patience, mode="max"),
        model_tag=f"{model_name}",
    )

    total_time = time.time() - t0
    best_idx = int(pd.Series(hist2["val_macro_f1"]).idxmax())
    best_val_acc = hist2["val_acc"][best_idx]
    best_val_f1 = hist2["val_macro_f1"][best_idx]

    return {
        "model": model_name,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_f1,
        "total_params": model.count_parameters()[0],
        "trainable_params": model.count_parameters()[1],
        "total_time_sec": round(total_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Suite")
    parser.add_argument("--model", default="efficientnetv2_s", choices=["efficientnetv2_s", "convnext_tiny", "mobilenetv3_large", "all"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger = Logger("transfer_learning", log_dir=cfg.paths.logs_dir)
    splits_dir = Path(cfg.paths.splits_dir)

    train_df = pd.read_csv(splits_dir / "plantvillage_train.csv")
    val_df = pd.read_csv(splits_dir / "plantvillage_val.csv")
    class_weights = compute_class_weights(train_df, num_classes=cfg.data.num_classes)

    models_to_train = [args.model] if args.model != "all" else ["efficientnetv2_s", "convnext_tiny", "mobilenetv3_large"]
    all_results = []

    for m in models_to_train:
        res = train_two_stage(m, cfg, train_df, val_df, class_weights, device, logger, image_size=cfg.data.image_size)
        all_results.append(res)

    out_path = Path(cfg.paths.reports_dir) / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    table = Table(title="Transfer Learning Performance")
    table.add_column("Model Architecture", style="cyan")
    table.add_column("Val Accuracy", style="magenta")
    table.add_column("Val Macro-F1", style="green")
    table.add_column("Total Parameters", style="yellow")
    table.add_column("Wall Time", style="blue")

    for r in all_results:
        table.add_row(
            r["model"],
            f"{r['val_acc']:.4f}",
            f"{r['val_macro_f1']:.4f}",
            f"{r['total_params']:,}",
            f"{r['total_time_sec']}s",
        )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
