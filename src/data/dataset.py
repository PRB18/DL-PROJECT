"""
src/data/dataset.py
-------------------
PyTorch Dataset and DataLoader factories for plant disease detection.

Design decisions
────────────────
1. The Dataset reads from a CSV manifest (path, label) instead of walking
   directories. This makes it trivial to swap in any split and prevents
   accidental directory-walk order variation across OSes.

2. Augmentations are Albumentations pipelines (not torchvision transforms).
   Why Albumentations? It's ~2–10× faster on CPU than torchvision for the same
   transforms, and it supports spatial transforms applied consistently to both
   image and mask (useful if we later add segmentation).

3. CutMix is applied at the collate level (not per-sample) so the mixed images
   share the same label distribution. We use a simple collate_fn wrapper.

4. Background randomisation: PlantVillage images have plain white/grey
   backgrounds. We detect near-white pixels (HSV S < 30 and V > 220) and
   replace them with a random crop from a background texture bank. This forces
   the model to learn leaf features rather than background colour.
   If no texture bank is available, we fall back to random noise.

5. Weighted sampler vs weighted loss: both address class imbalance. We
   support both and compare them in the ablation (Phase 3).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────────────────────

def get_train_transforms(
    image_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Strong augmentation for training (Albumentations ≥ 2.0 API).

    Each transform with its rationale:
    • RandomResizedCrop  — simulates different distances / zoom levels.
      NOTE: Albumentations 2.0 uses size=(H, W) instead of height/width.
    • HorizontalFlip     — leaves don't have a left/right preference.
    • Rotate(30°)        — robustness to phone orientation.
    • ColorJitter        — different lighting conditions on the farm.
    • GaussianBlur       — simulates out-of-focus / motion blur.
    • CLAHE              — enhances contrast for low-light photos.
    • GaussNoise         — adds sensor noise robustness.
      NOTE: Albumentations 2.0 uses std_range instead of var_limit.
    • CoarseDropout      — simulates partial occlusion (like CutOut).
      NOTE: Albumentations 2.0 uses num_holes_range/hole_height_range/fill.
    • Normalize + ToTensor — standard ImageNet normalisation.
    """
    hole_size = image_size // 8
    return A.Compose([
        # Albumentations ≥ 2.0: size=(height, width) replaces height= & width=
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1, p=0.6
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        # Albumentations ≥ 2.0: std_range replaces var_limit
        # std_range=(a, b) means std is sampled from [a, b]; equivalent
        # to old var_limit=(a², b²) but more intuitive
        A.GaussNoise(std_range=(0.03, 0.15), p=0.3),
        # Albumentations ≥ 2.0: num_holes_range, hole_height_range, fill
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(hole_size // 2, hole_size),
            hole_width_range=(hole_size // 2, hole_size),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Minimal transforms for validation and test — no data augmentation,
    only resize, crop, and normalise. This gives a fair, consistent evaluation.
    """
    return A.Compose([
        A.Resize(height=int(image_size * 1.14), width=int(image_size * 1.14)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
# Background randomisation
# ──────────────────────────────────────────────────────────────

def randomize_background(
    image: np.ndarray,
    texture_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Replace near-white/grey background pixels with a random texture or noise.

    Parameters
    ----------
    image       : H×W×3 uint8 RGB array.
    texture_dir : Directory of background JPEG/PNG textures. If None, uses noise.

    Returns
    -------
    Modified H×W×3 uint8 RGB array.

    Why?
    ────
    PlantVillage images are shot on plain backgrounds. A CNN can achieve high
    accuracy by keying off the background colour (white → healthy, grey →
    diseased). By randomising the background at training time, we force the
    model to actually look at the leaf texture and colour.
    """
    # Convert to HSV to detect near-neutral (grey/white) pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]  # Saturation
    v_channel = hsv[:, :, 2]  # Value (brightness)
    # Background: low saturation AND high brightness
    bg_mask = (s_channel < 30) & (v_channel > 200)

    if bg_mask.sum() < 100:
        return image  # No clear background — skip

    # Generate replacement texture
    h, w = image.shape[:2]
    if texture_dir is not None:
        textures = list(Path(texture_dir).glob("*.jpg")) + \
                   list(Path(texture_dir).glob("*.png"))
        if textures:
            tex_path = random.choice(textures)
            tex = np.array(Image.open(tex_path).convert("RGB").resize((w, h)))
        else:
            tex = _random_noise_bg(h, w)
    else:
        tex = _random_noise_bg(h, w)

    output = image.copy()
    output[bg_mask] = tex[bg_mask]
    return output


def _random_noise_bg(h: int, w: int) -> np.ndarray:
    """Return a smoothed Perlin-like noise texture as uint8 RGB."""
    noise = np.random.randint(80, 200, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for plant disease classification.

    Parameters
    ----------
    df              : DataFrame with columns ['path', 'label', 'class_name'].
    transform       : Albumentations Compose pipeline.
    background_rand : Whether to apply background randomisation.
    texture_dir     : Directory of background textures (optional).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        background_rand: bool = False,
        texture_dir: Optional[Path] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.background_rand = background_rand
        self.texture_dir = texture_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])

        # Load image
        image = np.array(Image.open(path).convert("RGB"))

        # Background randomisation (training only)
        if self.background_rand:
            image = randomize_background(image, self.texture_dir)

        # Albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]   # Tensor after ToTensorV2

        return {
            "image": image,
            "label": label,
            "path":  path,
            "class_name": row.get("class_name", ""),
        }


# ──────────────────────────────────────────────────────────────
# CutMix collate function
# ──────────────────────────────────────────────────────────────

def cutmix_collate(
    batch: list[dict],
    alpha: float = 1.0,
    prob: float = 0.3,
) -> dict[str, torch.Tensor | list]:
    """
    Custom collate function that applies CutMix with probability `prob`.

    CutMix vs Mixup
    ───────────────
    CutMix is preferred for classification of small lesion regions because it
    creates spatially coherent patches rather than pixel-level blending.
    The model still sees one coherent region of each class.

    Mixed labels are stored as (label_a, label_b, lam) for the loss function.
    """
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths  = [b["path"] for b in batch]

    if random.random() > prob:
        return {"image": images, "label": labels, "path": paths,
                "cutmix": False}

    # CutMix: generate random bounding box
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape

    # Sample random perm to get the "partner" images
    rand_idx = torch.randperm(B)

    # Get bounding box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    # Adjust lambda based on actual box area
    lam_actual = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)

    return {
        "image":    mixed,
        "label":    labels,
        "label_b":  labels[rand_idx],
        "lam":      lam_actual,
        "path":     paths,
        "cutmix":   True,
    }


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────

def build_dataloader(
    df: pd.DataFrame,
    split: str,                            # 'train' | 'val' | 'test'
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cutmix: bool = True,
    cutmix_prob: float = 0.3,
    use_weighted_sampler: bool = False,
    background_rand: bool = False,
    texture_dir: Optional[Path] = None,
    mean: tuple = (0.485, 0.456, 0.406),
    std:  tuple = (0.229, 0.224, 0.225),
) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Parameters
    ----------
    use_weighted_sampler : If True, use WeightedRandomSampler to balance classes
                           at the sampling level (alternative to weighted loss).
    background_rand      : Only applied during training.
    """
    is_train = (split == "train")

    transform = (
        get_train_transforms(image_size, mean, std)
        if is_train
        else get_val_transforms(image_size, mean, std)
    )

    dataset = PlantDiseaseDataset(
        df=df,
        transform=transform,
        background_rand=(is_train and background_rand),
        texture_dir=texture_dir,
    )

    # Collate function (CutMix only for training)
    collate_fn = None
    if is_train and use_cutmix:
        collate_fn = lambda b: cutmix_collate(b, prob=cutmix_prob)

    # Weighted sampler for class-imbalance handling
    sampler = None
    shuffle = is_train
    if is_train and use_weighted_sampler:
        class_counts = df["label"].value_counts().sort_index().values
        # Weight per class = inverse frequency
        weights_per_class = 1.0 / (class_counts + 1e-6)
        sample_weights = torch.tensor(
            [weights_per_class[label] for label in df["label"].values],
            dtype=torch.float,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
        drop_last=is_train,   # avoids tiny last batch destabilising BN
    )


# ──────────────────────────────────────────────────────────────
# Entry point — smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.seed import set_global_seed
    set_global_seed(42)

    # Quick smoke test with a dummy CSV
    import tempfile, os

    dummy_data = {
        "path": ["data/raw/PlantVillage/Tomato___healthy/image_0001.JPG"] * 10,
        "class_name": ["Tomato___healthy"] * 10,
        "label": [0] * 10,
    }
    df = pd.DataFrame(dummy_data)

    # Only test if the file actually exists
    existing = df[df["path"].apply(lambda p: Path(p).exists())]
    if existing.empty:
        print("No PlantVillage images found. Download the dataset first.")
    else:
        loader = build_dataloader(existing, split="train", batch_size=4, num_workers=0)
        batch = next(iter(loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch labels:      {batch['label']}")
        print("✅  DataLoader smoke test passed.")
