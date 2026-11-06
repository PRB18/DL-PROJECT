"""
src/explainability/gradcam.py
-----------------------------
Grad-CAM and Grad-CAM++ Visual Explanations for Plant Disease Diagnosis:
- Implements both vanilla Grad-CAM and Grad-CAM++ (Chattopadhyay et al., 2018).
- Automatically targets final convolutional feature extraction layers across backbones.
- Measures Background vs Foreground Attention Ratio (detects lab-background bias).
- Generates side-by-side diagnostic figures: [Original | Heatmap | Overlay | Diagnosis Info].
- Formulates automated hypotheses for worst misclassifications.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Finds the last convolutional layer of the model architecture."""
    if hasattr(model, "backbone"):
        backbone = model.backbone
        # ConvNeXt
        if hasattr(backbone, "stages"):
            return backbone.stages[-1].blocks[-1]
        # EfficientNet / MobileNet
        if hasattr(backbone, "blocks"):
            return backbone.blocks[-1][-1]
        # ResNet
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1]

    # Fallback for baseline CNN or arbitrary model: locate last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError(f"Could not automatically determine target layer for {model_name}")


class GradCAMPlusPlus:
    """
    Native PyTorch Grad-CAM++ engine.
    Uses second- and third-order gradients to sharpen localization on multiple lesion spots.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register forward and backward hooks
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0] # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # Grad-CAM++ weighting coefficients
        # alpha_k^c = (grad^2) / (2 * grad^2 + sum(act * grad^3) + eps)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        eps = 1e-7

        sum_act_grad3 = torch.sum(activations * grad_3, dim=(1, 2), keepdim=True)
        alpha = grad_2 / (2.0 * grad_2 + sum_act_grad3 + eps)

        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()
        return cam_np

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def overlay_heatmap(
    original_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlays resized heatmap on the RGB image."""
    h, w = original_img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * original_img_rgb)
    return blended


def analyze_background_attention(
    model: nn.Module,
    target_layer: nn.Module,
    img_rgb: np.ndarray,
    transform,
    device: torch.device,
) -> Dict[str, float]:
    """
    Computes ratio of heatmap intensity in the plain background vs leaf foreground.
    """
    # 1. Background mask via HSV thresholding
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    bg_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
    fg_mask = ~bg_mask

    # 2. Grad-CAM++ heatmap
    engine = GradCAMPlusPlus(model, target_layer)
    inp = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    cam = engine.generate(inp)
    engine.remove_hooks()

    cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))

    bg_mean = float(np.mean(cam_resized[bg_mask])) if np.sum(bg_mask) > 50 else 0.0
    fg_mean = float(np.mean(cam_resized[fg_mask])) if np.sum(fg_mask) > 50 else 1.0

    ratio = bg_mean / (fg_mean + 1e-6)
    return {
        "bg_attention": bg_mean,
        "fg_attention": fg_mean,
        "bg_to_fg_ratio": ratio,
        "is_background_biased": bool(ratio > 0.8),
    }


def create_explanation_figure(
    img_rgb: np.ndarray,
    cam: np.ndarray,
    true_label_name: str,
    pred_label_name: str,
    confidence: float,
    output_path: Path,
) -> None:
    """Generates 4-panel diagnostic explanation figure."""
    overlay = overlay_heatmap(img_rgb, cam)
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Leaf Photo")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM++ Lesion Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Diagnostic Overlay")
    axes[2].axis("off")

    is_correct = (true_label_name == pred_label_name)
    color = "green" if is_correct else "red"
    status = "CORRECT" if is_correct else "MISCLASSIFIED"

    axes[3].text(0.1, 0.8, f"Status: {status}", color=color, weight="bold", fontsize=14)
    axes[3].text(0.1, 0.6, f"True: {true_label_name[:25]}", fontsize=11)
    axes[3].text(0.1, 0.45, f"Pred: {pred_label_name[:25]}", fontsize=11)
    axes[3].text(0.1, 0.3, f"Confidence: {confidence * 100:.1f}%", fontsize=12, weight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
