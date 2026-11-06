"""
src/data/class_weights.py
--------------------------
Compute inverse-frequency class weights for the weighted cross-entropy loss.

Design decision
───────────────
PlantVillage has mild imbalance (ratio ~6:1). Two strategies:
  A) Weighted loss: scale the CE loss per class by inverse frequency.
     Pros: simple, doesn't change the data distribution.
     Cons: doesn't help the model "see" minority classes more often.

  B) Weighted sampler: oversample minority classes at the batch level.
     Pros: model sees balanced batches.
     Cons: majority classes get seen less; can overfit to minority.

We implement both and compare in Phase 3 ablation. The weights computed here
are used in both strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int,
    method: str = "inverse_freq",
    smoothing: float = 1e-6,
) -> torch.Tensor:
    """
    Compute class weights for the CE loss.

    Parameters
    ----------
    train_df    : Training split DataFrame with a 'label' column (int indices).
    num_classes : Total number of classes.
    method      : 'inverse_freq' | 'sqrt_inverse' | 'effective_num'.
    smoothing   : Small constant to avoid division by zero.

    Returns
    -------
    1-D FloatTensor of shape (num_classes,), normalised so the median = 1.0.
    """
    counts = train_df["label"].value_counts().sort_index()
    # Ensure all classes are represented (some may have 0 training samples)
    counts = counts.reindex(range(num_classes), fill_value=smoothing)
    counts_arr = counts.values.astype(np.float64)

    if method == "inverse_freq":
        # Weight ∝ 1/count → common classes penalised less
        weights = 1.0 / (counts_arr + smoothing)

    elif method == "sqrt_inverse":
        # Softer than inverse_freq; empirically better for mild imbalance
        weights = 1.0 / np.sqrt(counts_arr + smoothing)

    elif method == "effective_num":
        # From: "Class-Balanced Loss Based on Effective Number of Samples"
        # Cui et al., CVPR 2019
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / (effective_num + smoothing)

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose from: inverse_freq, sqrt_inverse, effective_num")

    # Normalise: median weight = 1.0  (keeps the overall loss scale stable)
    weights = weights / (np.median(weights) + smoothing)
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Quick demo
    dummy_df = pd.DataFrame({
        "label": [0]*100 + [1]*50 + [2]*10 + [3]*200,
    })
    for method in ["inverse_freq", "sqrt_inverse", "effective_num"]:
        w = compute_class_weights(dummy_df, num_classes=4, method=method)
        print(f"{method}: {w.numpy().round(3)}")
