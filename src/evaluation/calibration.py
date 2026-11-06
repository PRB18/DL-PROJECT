"""
src/evaluation/calibration.py
-----------------------------
Model probability calibration and Temperature Scaling.

Theoretical Formulation:
-------------------------
Deep convolutional neural networks trained with cross-entropy and high accuracy
are frequently overconfident: a model outputting 0.99 softmax probability may be
correct only 75% of the time. In agricultural disease diagnosis, false certainty
leads to incorrect chemical pesticide application.

We quantify calibration using Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

Temperature Scaling (Guo et al., 2017):
Divides raw network logits z by a learned temperature scalar T > 0:
    p_i = exp(z_i / T) / sum_j exp(z_j / T)
T > 1 softens overconfident probabilities without changing the argmax prediction
(preserving Top-1 and Top-k accuracy identically).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error across n_bins.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
) -> None:
    """Plots confidence vs accuracy calibration curve."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_centers[i])

    ece = compute_ece(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, color="royalblue", edgecolor="black", label="Model Outputs")
    ax.plot([0, 1], [0, 1], "--", color="red", linewidth=2, label="Perfect Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title} (ECE = {ece * 100:.2f}%)", fontsize=14)
    ax.legend(loc="upper left")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling layer.
    Optimizes single temperature scalar T via L-BFGS on validation logits.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_loader: DataLoader, model: nn.Module, device: torch.device) -> "TemperatureScaler":
        model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                logits_list.append(logits)
                labels_list.append(labels)

        logits_all = torch.cat(logits_list).to(device)
        labels_all = torch.cat(labels_list).to(device)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self
