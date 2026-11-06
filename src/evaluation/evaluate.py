"""
src/evaluation/evaluate.py
--------------------------
Evaluation pipeline orchestrator:
1. Computes all standard classification metrics on test split.
2. Plots and saves Confusion Matrix and Per-Class F1 bars.
3. Performs Reliability Diagram analysis and Temperature Scaling calibration.
4. Executes environmental robustness suite.
5. Performs domain shift evaluation against PlantDoc.
6. Exports report to reports/evaluation_report.json.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import build_dataloader
from src.data.domain_shift import run_domain_shift_evaluation
from src.evaluation.calibration import (
    TemperatureScaler,
    compute_ece,
    plot_reliability_diagram,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    print_metrics_table,
)
from src.evaluation.robustness import (
    plot_robustness_curves,
    run_all_robustness_tests,
)
from src.models.transfer import build_model
from src.utils.config import load_config
from src.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection Model")
    parser.add_argument("--checkpoint", default="checkpoints/best_efficientnetv2_s.pth")
    parser.add_argument("--model-name", default="efficientnetv2_s")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = Logger("evaluator", log_dir=cfg.paths.logs_dir)

    # Load class map
    class_map_path = Path(cfg.paths.splits_dir) / "plantvillage_class_map.json"
    with open(class_map_path) as f:
        class_map = json.load(f)
    idx_to_class = class_map["idx_to_class"]
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    test_csv = Path(cfg.paths.splits_dir) / "plantvillage_test.csv"
    val_csv = Path(cfg.paths.splits_dir) / "plantvillage_val.csv"
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)

    test_loader = build_dataloader(
        df=test_df,
        split="test",
        image_size=cfg.data.image_size,
        batch_size=32,
        num_workers=cfg.data.num_workers,
    )
    val_loader = build_dataloader(
        df=val_df,
        split="val",
        image_size=cfg.data.image_size,
        batch_size=32,
        num_workers=cfg.data.num_workers,
    )

    # Initialize model
    model = build_model(args.model_name, num_classes=cfg.data.num_classes, pretrained=False)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("state_dict", state))
        logger.info(f"Loaded model weights from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Running with initialized weights.")
    model.to(device)
    model.eval()

    # 1. Inference on test set
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["label"].numpy()
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets)

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    probs_arr = np.exp(logits_arr) / np.sum(np.exp(logits_arr), axis=1, keepdims=True)
    preds_arr = np.argmax(probs_arr, axis=1)

    # 2. Compute all metrics
    metrics = compute_all_metrics(targets_arr, preds_arr, probs_arr, class_names, cfg.data.num_classes)
    print_metrics_table(metrics, args.model_name)

    # 3. Save Figures
    figures_dir = Path(cfg.paths.reports_dir) / "figures"
    plot_confusion_matrix(targets_arr, preds_arr, class_names, figures_dir / "confusion_matrix.png")
    plot_per_class_metrics(metrics["per_class_report"], figures_dir / "per_class_f1.png")
    plot_reliability_diagram(targets_arr, probs_arr, figures_dir / "calibration_before.png", title="Calibration Before Scaling")

    # 4. Temperature Scaling
    scaler = TemperatureScaler().fit(val_loader, model, device)
    temp_val = float(scaler.temperature.item())
    logger.info(f"Learned Optimal Temperature: T = {temp_val:.4f}")

    scaled_probs = np.exp(logits_arr / temp_val) / np.sum(np.exp(logits_arr / temp_val), axis=1, keepdims=True)
    plot_reliability_diagram(targets_arr, scaled_probs, figures_dir / "calibration_after.png", title="Calibration After Temp Scaling")

    # 5. Domain Shift against PlantDoc
    plantdoc_csv = Path(cfg.paths.splits_dir) / "plantdoc_test.csv"
    plantdoc_df = pd.read_csv(plantdoc_csv) if plantdoc_csv.exists() else None
    run_domain_shift_evaluation(
        model=model,
        pv_test_df=test_df,
        plantdoc_df=plantdoc_df,
        class_map_path=class_map_path,
        image_size=cfg.data.image_size,
        device=device,
        output_dir=Path(cfg.paths.reports_dir),
    )

    # 6. Save JSON report
    report_out = Path(cfg.paths.reports_dir) / "evaluation_report.json"
    with open(report_out, "w") as f:
        json.dump({
            "model": args.model_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "top3_accuracy": metrics["top3_accuracy"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "ece_before": compute_ece(targets_arr, probs_arr),
            "ece_after": compute_ece(targets_arr, scaled_probs),
            "temperature": temp_val,
        }, f, indent=2)

    logger.info(f"Full evaluation report saved to {report_out}")


if __name__ == "__main__":
    main()
