"""
src/training/scheduler.py
-------------------------
Learning rate scheduler builders.

Design Decisions & Viva Defense:
--------------------------------
1. Why Cosine Annealing with Warmup?
   - At the beginning of training, weights in the classification head are initialized randomly.
     Large initial gradients from random predictions can destabilize the early representation layers.
   - Linear warmup starts at 1% of the peak learning rate and gradually increases to 100% over the
     warmup period (e.g. 2-3 epochs).
   - Once warmup completes, Cosine Annealing smoothly decays learning rates down to a small floor,
     allowing the optimizer to settle into flatter, more generalizable local minima without sharp drops.
"""

from typing import Any, Dict
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    warmup_epochs: int = 2,
    total_epochs: int = 30,
    min_lr: float = 1e-6,
) -> Dict[str, Any]:
    """
    Constructs an epoch-stepped LR scheduler.

    Returns:
        dict: {"scheduler": lr_scheduler, "interval": "epoch"}
    """
    if scheduler_type == "cosine_warmup" and warmup_epochs > 0:
        if total_epochs <= warmup_epochs:
            # Degenerate case: fallback to pure linear warmup
            sched = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=total_epochs,
            )
        else:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )
            sched = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
    elif scheduler_type in ["cosine", "cosine_warmup"]:
        sched = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return {"scheduler": sched, "interval": "epoch"}


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Extract current learning rate from optimizer's first parameter group."""
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0
