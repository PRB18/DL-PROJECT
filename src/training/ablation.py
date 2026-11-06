"""
src/training/ablation.py
------------------------
Ablation study matrix and runner:
Investigates the individual and joint contributions of:
1. Data Augmentation (Baseline vs Albumentations with CutMix + Background Randomization)
2. Fine-tuning strategy (Frozen Feature Extraction vs Full Fine-Tuning with Discriminative LRs)
3. Input Resolution (224x224 vs 288x288)
4. Class Imbalance Mitigation (Unweighted vs Inverse-Frequency Class Weighted Loss)
5. Multi-seed stability (Seeds: 42, 123, 7)
"""

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import torch

from src.data.class_weights import compute_class_weights
from src.data.dataset import build_dataloader
from src.models.transfer import build_model
from src.training.losses import LabelSmoothingCE
from src.training.scheduler import build_scheduler
from src.training.trainer import EarlyStopping, Trainer
from src.utils.config import load_config
from src.utils.logger import Logger
from src.utils.seed import set_global_seed


@dataclass
class AblationConfig:
    seed: int
    image_size: int
    use_augmentation: bool
    fine_tune: bool
    use_class_weighting: bool


def run_single_ablation(
    model_name: str,
    cfg,
    ablation_cfg: AblationConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    logger: Logger,
) -> Dict[str, float]:
    set_global_seed(ablation_cfg.seed)

    # Dataloader with or without augmentations
    train_loader = build_dataloader(
        df=train_df,
        split="train" if ablation_cfg.use_augmentation else "val", # val split has no random augs
        image_size=ablation_cfg.image_size,
        batch_size=32,
        num_workers=cfg.data.num_workers,
        use_cutmix=ablation_cfg.use_augmentation,
        background_rand=ablation_cfg.use_augmentation,
    )
    val_loader = build_dataloader(
        df=val_df,
        split="val",
        image_size=ablation_cfg.image_size,
        batch_size=32,
        num_workers=cfg.data.num_workers,
    )

    class_weights = None
    if ablation_cfg.use_class_weighting:
        class_weights = compute_class_weights(train_df, num_classes=cfg.data.num_classes)

    criterion = LabelSmoothingCE(
        num_classes=cfg.data.num_classes,
        smoothing=0.1,
        weight=class_weights,
    )

    model = build_model(
        model_name,
        num_classes=cfg.data.num_classes,
        pretrained=True,
        frozen=not ablation_cfg.fine_tune,
    )

    if ablation_cfg.fine_tune:
        params = model.get_param_groups(backbone_lr_factor=0.1, head_lr=1e-4)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, "cosine_warmup", warmup_epochs=1, total_epochs=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        cfg=cfg,
        logger=logger,
        checkpoint_dir=Path("checkpoints/ablations"),
        use_amp=cfg.project.mixed_precision,
    )

    tag = f"ablation_{model_name}_s{ablation_cfg.seed}_res{ablation_cfg.image_size}_aug{ablation_cfg.use_augmentation}"
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        early_stopping=EarlyStopping(patience=3),
        model_tag=tag,
    )

    best_idx = int(pd.Series(history["val_macro_f1"]).idxmax())
    return {
        "val_acc": history["val_acc"][best_idx],
        "val_macro_f1": history["val_macro_f1"][best_idx],
    }


def main():
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger("ablation_study", log_dir=cfg.paths.logs_dir)

    train_df = pd.read_csv("data/splits/plantvillage_train.csv")
    val_df = pd.read_csv("data/splits/plantvillage_val.csv")

    # Define core informative ablation settings
    experiments = [
        # 1. Baseline: 224, with aug, fine-tuned, with weights (Control across 3 seeds)
        AblationConfig(seed=42, image_size=224, use_augmentation=True, fine_tune=True, use_class_weighting=True),
        AblationConfig(seed=123, image_size=224, use_augmentation=True, fine_tune=True, use_class_weighting=True),
        AblationConfig(seed=7, image_size=224, use_augmentation=True, fine_tune=True, use_class_weighting=True),
        # 2. No Augmentation
        AblationConfig(seed=42, image_size=224, use_augmentation=False, fine_tune=True, use_class_weighting=True),
        # 3. Frozen Backbone only
        AblationConfig(seed=42, image_size=224, use_augmentation=True, fine_tune=False, use_class_weighting=True),
        # 4. High resolution 288x288
        AblationConfig(seed=42, image_size=288, use_augmentation=True, fine_tune=True, use_class_weighting=True),
        # 5. Without class weighting
        AblationConfig(seed=42, image_size=224, use_augmentation=True, fine_tune=True, use_class_weighting=False),
    ]

    records = []
    for exp in experiments:
        logger.info(f"\nRunning ablation config: {exp}")
        res = run_single_ablation("efficientnetv2_s", cfg, exp, train_df, val_df, device, logger)
        rec = {
            "seed": exp.seed,
            "image_size": exp.image_size,
            "augmentation": exp.use_augmentation,
            "fine_tune": exp.fine_tune,
            "class_weighting": exp.use_class_weighting,
            "val_acc": res["val_acc"],
            "val_macro_f1": res["val_macro_f1"],
        }
        records.append(rec)

    res_df = pd.DataFrame(records)
    out_csv = Path(cfg.paths.reports_dir) / "ablation_results.csv"
    res_df.to_csv(out_csv, index=False)
    logger.info(f"Ablation results saved -> {out_csv}")


if __name__ == "__main__":
    main()
