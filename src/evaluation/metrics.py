"""
src/evaluation/metrics.py
-------------------------
Comprehensive evaluation suite for plant disease classification:
- Multiclass Accuracy, Macro-F1, Weighted-F1
- Top-k Accuracy (Top-3)
- Multiclass One-vs-Rest ROC-AUC
- Row-normalized Confusion Matrix with Seaborn
- Per-class F1 horizontal bar chart with performance threshold coloring
- Rich table visualization
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """Computes whether true label is within top-k highest probability classes."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    hits = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(np.mean(hits))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    num_classes: int = 38,
) -> Dict[str, Union[float, dict, list]]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    top3_acc = top_k_accuracy(y_true, y_proba, k=min(3, num_classes))

    # One-vs-Rest ROC-AUC
    try:
        roc_auc_ovr = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc_ovr = 0.0

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_accuracy": top3_acc,
        "roc_auc_ovr": roc_auc_ovr,
        "per_class_report": report,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    figsize: Tuple[int, int] = (18, 16),
) -> plt.Figure:
    """Plots and saves normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.split("___")[-1][:12] for c in class_names],
        yticklabels=[c.split("___")[-1][:12] for c in class_names],
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix (Row Recall)", fontsize=16)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return fig


def plot_per_class_metrics(
    per_class_report: dict,
    output_path: Path,
    min_f1_threshold: float = 0.8,
) -> None:
    """Horizontal bar chart coloring classes: Green (>0.9), Yellow (>0.7), Red (<=0.7)."""
    classes = [k for k in per_class_report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    f1_scores = [per_class_report[c]["f1-score"] for c in classes]

    # Sort descending
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1])
    sorted_classes = [p[0].split("___")[-1] for p in sorted_pairs]
    sorted_f1s = [p[1] for p in sorted_pairs]

    colors = []
    for f in sorted_f1s:
        if f >= 0.90:
            colors.append("#2ecc71") # Green
        elif f >= 0.70:
            colors.append("#f39c12") # Yellow/Orange
        else:
            colors.append("#e74c3c") # Red

    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_classes) * 0.3)))
    bars = ax.barh(sorted_classes, sorted_f1s, color=colors)
    ax.axvline(0.9, color="green", linestyle="--", alpha=0.5, label="High Perf (0.90)")
    ax.axvline(0.7, color="orange", linestyle="--", alpha=0.5, label="Acceptable (0.70)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score")
    ax.set_title("Per-Class F1 Score Distribution")
    ax.legend(loc="lower right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_metrics_table(metrics_dict: dict, model_name: str) -> None:
    """Print high-level metrics table via Rich."""
    table = Table(title=f"Evaluation Metrics: {model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green" if metrics_dict["macro_f1"] >= 0.9 else "yellow")

    table.add_row("Top-1 Accuracy", f"{metrics_dict['accuracy'] * 100:.2f}%")
    table.add_row("Macro F1", f"{metrics_dict['macro_f1']:.4f}")
    table.add_row("Weighted F1", f"{metrics_dict['weighted_f1']:.4f}")
    table.add_row("Top-3 Accuracy", f"{metrics_dict['top3_accuracy'] * 100:.2f}%")
    table.add_row("Macro ROC-AUC (OvR)", f"{metrics_dict['roc_auc_ovr']:.4f}")

    console = Console()
    console.print(table)
