"""
src/training/losses.py
----------------------
Custom loss functions designed for plant disease classification.

Includes:
1. LabelSmoothingCE: Prevents network overconfidence and aids probability calibration.
2. compute_cutmix_loss: Supports dual-label convex combination for CutMix samples.
3. FocalLoss: Addresses acute class imbalances by down-weighting well-classified easy samples.

Design Rationale:
-----------------
- Label Smoothing: Standard CrossEntropy forces logits for the true class toward infinity,
  which exacerbates model miscalibration. By softening the ground truth target from 1.0 to
  (1.0 - smoothing + smoothing / K) and spreading (smoothing / K) across remaining classes,
  the representations become more clustered and less overconfident.
- CutMix Loss: When a sample contains a composite image of class A and class B with area ratio λ,
  the objective function must evaluate loss linearly as:
    L_mixed = λ * L(pred, y_a) + (1 - λ) * L(pred, y_b)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing and optional per-class weighting.
    """

    def __init__(
        self,
        num_classes: int = 38,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes) unnormalized predictions
        targets: (B,) integer class labels or (B, num_classes) soft probabilities
        """
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            # One-hot encode with label smoothing
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Targets are already soft distributions (e.g. from mixup)
            true_dist = targets

        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = -(true_dist * log_probs * weight.unsqueeze(0)).sum(dim=-1)
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)

        return loss.mean()


def compute_cutmix_loss(
    logits: torch.Tensor,
    label_a: torch.Tensor,
    label_b: torch.Tensor,
    lam: float,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    Computes weighted loss for CutMix batch:
    L = lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)
    """
    loss_a = criterion(logits, label_a)
    loss_b = criterion(logits, label_b)
    return lam * loss_a + (1.0 - lam) * loss_b


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    Down-weights easy examples and focuses training on ambiguous / minority classes.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)

        # Gather probability of true classes
        if targets.dim() == 1:
            log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Soft targets
            log_pt = (targets * log_pt).sum(dim=-1)
            pt = (targets * pt).sum(dim=-1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            if targets.dim() == 1:
                at = self.alpha.gather(0, targets)
            else:
                at = (targets * self.alpha).sum(dim=-1)
            loss = loss * at

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
