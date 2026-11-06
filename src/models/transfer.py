"""
src/models/transfer.py
----------------------
Transfer learning model factory supporting:
- ResNet50 (Baseline B and full fine-tune)
- EfficientNetV2-S (High efficiency, compound scaling)
- ConvNeXt-Tiny (Modernized pure-convolutional vision architecture)
- MobileNetV3-Large (Ultra-lightweight edge deployment backbone)

Design Decisions & Technical Justifications:
--------------------------------------------
1. Why timm (PyTorch Image Models)?
   - Industry-standard model repository with battle-tested implementations and accurate pretrained weights.
   - Clean uniform interface: `num_classes=0` cleanly strips the classification head and returns raw features.
2. Two-Stage Fine-Tuning Strategy:
   - Stage 1 (Head-only): Backbone weights are frozen (`requires_grad = False`). Only the newly initialized linear
     projection layer receives gradients. This protects pretrained ImageNet representation features from getting
     destroyed by large initial gradients caused by random head initialization.
   - Stage 2 (Full fine-tune): All layers are unfrozen, but with Discriminative Learning Rates (e.g. 10x smaller learning
     rate for the backbone than the head). Early convolutional layers learn universal primitive edges and textures that
     rarely need large shifts, while deeper layers adapt to high-level botanical pathogen manifestations.
3. Feature Pooling & Classifier:
   - Dropout (0.2) + Linear(feat_dim, num_classes).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

SUPPORTED_MODELS: Dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "mobilenetv3_large": "mobilenetv3_large_100",
}


class TransferModel(nn.Module):
    """
    Unified transfer learning wrapper around timm backbones.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 38,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("The 'timm' package is required. Install via `pip install timm`.")

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. Choose from {list(SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.timm_id = SUPPORTED_MODELS[model_name]
        self.num_classes = num_classes

        # Create backbone without classification head (pooling still preserved or output flattened)
        create_kwargs = {
            "model_name": self.timm_id,
            "pretrained": pretrained,
            "num_classes": 0,
        }
        # Only pass drop_path_rate if model architecture supports stochastic depth
        if model_name in ["efficientnetv2_s", "convnext_tiny"]:
            create_kwargs["drop_path_rate"] = drop_path_rate

        self.backbone = timm.create_model(**create_kwargs)

        # Determine feature dimension from the backbone
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            feat_dim = self.backbone.head.fc.in_features
        else:
            # Fallback probe with dummy tensor
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feat = self.backbone(dummy)
                feat_dim = feat.shape[1]

        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        logits = self.head(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Stage 1 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for Stage 2 fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        backbone_lr_factor: float = 0.1,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> List[Dict]:
        """
        Configure discriminative learning rate parameter groups.
        Backbone gets (head_lr * backbone_lr_factor); Head gets head_lr.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        return [
            {
                "params": backbone_params,
                "lr": head_lr * backbone_lr_factor,
                "weight_decay": weight_decay,
            },
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(
    model_name: str,
    num_classes: int = 38,
    pretrained: bool = True,
    frozen: bool = False,
    drop_rate: float = 0.2,
) -> nn.Module:
    """
    Factory function for both baseline and transfer learning architectures.
    """
    if model_name == "baseline_cnn":
        from src.models.baseline_cnn import build_baseline_cnn
        return build_baseline_cnn(num_classes=num_classes)

    model = TransferModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
    if frozen:
        model.freeze_backbone()
    return model
