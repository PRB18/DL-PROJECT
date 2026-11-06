"""
src/data/domain_shift.py
------------------------
Domain-shift evaluation: train on PlantVillage, evaluate on PlantDoc.

This script loads a trained model and reports:
  1. Accuracy, macro-F1, per-class F1 on the PlantVillage TEST set.
  2. The same metrics on the PlantDoc matched subset (overlapping classes only).
  3. The "domain gap" = PV_test_accuracy − PlantDoc_accuracy.
  4. A per-class drop table (which classes suffer most on real photos).
  5. A visualisation: scatter plot of PV accuracy vs PlantDoc accuracy per class.

Design note
───────────
We evaluate only on OVERLAPPING classes. If PlantDoc has 20 classes and
PlantVillage has 38, we restrict to the 20 overlapping ones for a fair
comparison. This is explicitly flagged in the results.

Assumption / risk
─────────────────
PlantDoc class name matching to PlantVillage is done by fuzzy string matching.
Human review of the mapping is recommended before citing the numbers.
The mapping is logged to data/splits/class_name_mapping.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PlantDiseaseDataset, get_val_transforms


# ──────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 64,
    device: str = "cuda",
    label_col: str = "label",
) -> dict:
    """
    Run inference on a DataFrame split and return metrics.

    Returns
    -------
    dict with keys: accuracy, macro_f1, weighted_f1, per_class_report, preds, targets.
    """
    model.eval()
    transform = get_val_transforms(image_size)
    dataset = PlantDiseaseDataset(df=df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images  = batch["image"].to(device)
        targets = batch["label"]

        logits = model(images)
        preds  = logits.argmax(dim=-1).cpu()

        all_preds.extend(preds.numpy())
        all_targets.extend(targets.numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "accuracy":       float(accuracy_score(all_targets, all_preds)),
        "macro_f1":       float(f1_score(all_targets, all_preds, average="macro",   zero_division=0)),
        "weighted_f1":    float(f1_score(all_targets, all_preds, average="weighted", zero_division=0)),
        "per_class_report": classification_report(all_targets, all_preds, zero_division=0),
        "preds":   all_preds,
        "targets": all_targets,
    }


# ──────────────────────────────────────────────────────────────
# Domain-shift comparison
# ──────────────────────────────────────────────────────────────

def run_domain_shift_evaluation(
    model: torch.nn.Module,
    pv_test_df: pd.DataFrame,
    plantdoc_df: Optional[pd.DataFrame],
    class_map_path: Path,
    image_size: int = 224,
    device: str = "cuda",
    output_dir: Path = Path("reports"),
) -> None:
    """
    Compare model performance on PlantVillage (clean) vs PlantDoc (real-world).

    Parameters
    ----------
    model         : Trained model in eval mode.
    pv_test_df    : PlantVillage test split (with 'label' column).
    plantdoc_df   : PlantDoc split with 'pv_class' and 'label' columns.
                    If None, only PV evaluation is done.
    class_map_path: Path to plantvillage_class_map.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(class_map_path) as f:
        class_map = json.load(f)
    idx_to_class = class_map["idx_to_class"]

    print("\n" + "="*60)
    print("  DOMAIN SHIFT EVALUATION")
    print("="*60)

    # ── PlantVillage test set ──────────────────────────────────
    print("\n[1] PlantVillage TEST set (lab-style, clean backgrounds)")
    pv_metrics = evaluate_on_split(model, pv_test_df, image_size, device=device)
    print(f"    Accuracy:  {pv_metrics['accuracy']:.4f}")
    print(f"    Macro F1:  {pv_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {pv_metrics['weighted_f1']:.4f}")

    if plantdoc_df is None:
        print("\n[2] PlantDoc not available. Skipping domain-shift comparison.")
        return

    # ── PlantDoc (domain shift) ───────────────────────────────
    print("\n[2] PlantDoc TEST set (in-the-wild, real farm photos)")

    # Restrict to overlapping classes and re-encode labels to PV indices
    plantdoc_df = plantdoc_df.copy()
    class_to_idx = class_map["class_to_idx"]
    plantdoc_df["label"] = plantdoc_df["pv_class"].map(class_to_idx)
    plantdoc_df = plantdoc_df[plantdoc_df["label"].notna()].copy()
    plantdoc_df["label"] = plantdoc_df["label"].astype(int)

    pd_metrics = evaluate_on_split(model, plantdoc_df, image_size, device=device)
    print(f"    Accuracy:  {pd_metrics['accuracy']:.4f}")
    print(f"    Macro F1:  {pd_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {pd_metrics['weighted_f1']:.4f}")

    # ── Domain gap ───────────────────────────────────────────
    gap_acc = pv_metrics["accuracy"] - pd_metrics["accuracy"]
    gap_f1  = pv_metrics["macro_f1"] - pd_metrics["macro_f1"]

    print(f"\n[3] Domain Gap (PlantVillage → PlantDoc)")
    print(f"    Accuracy drop: {gap_acc:+.4f} ({gap_acc*100:+.1f} pp)")
    print(f"    Macro F1 drop: {gap_f1:+.4f} ({gap_f1*100:+.1f} pp)")
    print()
    if gap_acc > 0.20:
        print("    ⚠️  LARGE domain gap detected (>20 pp). The model may be")
        print("       learning background cues rather than leaf lesions.")
        print("       Recommendation: Use background randomisation augmentation.")

    # ── Per-class comparison plot ─────────────────────────────
    # Compute per-class accuracy for both sets on shared classes
    shared_classes = sorted(plantdoc_df["label"].unique())
    pv_subset_df   = pv_test_df[pv_test_df["label"].isin(shared_classes)]
    pv_sub_metrics = evaluate_on_split(model, pv_subset_df, image_size, device=device)

    # Per-class F1 for shared classes
    from sklearn.metrics import f1_score as sk_f1
    pv_per_class  = sk_f1(pv_sub_metrics["targets"], pv_sub_metrics["preds"],
                           labels=shared_classes, average=None, zero_division=0)
    pd_per_class  = sk_f1(pd_metrics["targets"], pd_metrics["preds"],
                           labels=shared_classes, average=None, zero_division=0)

    class_names = [idx_to_class[str(c)] for c in shared_classes]
    comparison_df = pd.DataFrame({
        "class":      class_names,
        "pv_f1":      pv_per_class,
        "plantdoc_f1": pd_per_class,
        "drop":       pv_per_class - pd_per_class,
    }).sort_values("drop", ascending=False)

    print("\n[4] Per-Class F1 Drop (top 10 worst):")
    print(comparison_df.head(10).to_string(index=False))

    # Save comparison
    out_csv = output_dir / "domain_shift_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\nFull comparison saved → {out_csv}")

    # ── Scatter plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        comparison_df["pv_f1"],
        comparison_df["plantdoc_f1"],
        c=comparison_df["drop"],
        cmap="RdYlGn_r",
        s=80, alpha=0.8, edgecolors="black", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="F1 drop (PV → PlantDoc)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect transfer (no gap)")

    for _, row in comparison_df.iterrows():
        ax.annotate(
            row["class"].split("___")[-1][:15],  # short name
            (row["pv_f1"], row["plantdoc_f1"]),
            fontsize=6, alpha=0.7
        )

    ax.set_xlabel("F1 on PlantVillage (clean, lab-style)")
    ax.set_ylabel("F1 on PlantDoc (real farm photos)")
    ax.set_title("Domain Shift: PlantVillage → PlantDoc\n"
                 "(Points below the dashed line = performance degradation)")
    ax.legend()
    plt.tight_layout()

    fig_path = output_dir / "figures" / "domain_shift_scatter.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved → {fig_path}")

    # ── Summary table ─────────────────────────────────────────
    summary = {
        "plantvillage_accuracy": pv_metrics["accuracy"],
        "plantvillage_macro_f1": pv_metrics["macro_f1"],
        "plantdoc_accuracy":     pd_metrics["accuracy"],
        "plantdoc_macro_f1":     pd_metrics["macro_f1"],
        "accuracy_gap_pp":       float(gap_acc * 100),
        "macro_f1_gap_pp":       float(gap_f1  * 100),
        "overlapping_classes":   len(shared_classes),
    }
    with open(output_dir / "domain_shift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved → {output_dir / 'domain_shift_summary.json'}")
