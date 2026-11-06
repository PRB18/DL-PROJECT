"""
src/evaluation/robustness.py
----------------------------
Real-world farm stress testing:
Simulates camera and environmental corruptions:
1. Blur (Defocus / Camera motion)
2. Gaussian Sensor Noise
3. WhatsApp / Low-bandwidth JPEG Compression
4. Under-exposure / Deep Shade (Low Light)
5. Arbitrary Camera Tilt / Rotation
6. Leaf Occlusion / Debris Overlap
7. Out-of-Distribution (OOD) non-leaf rejection based on max-softmax uncertainty.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def corrupt_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = [3, 5, 7, 9, 11][severity - 1]
    return cv2.GaussianBlur(img, (k, k), 0)


def corrupt_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = [5, 15, 30, 50, 80][severity - 1]
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    corrupted = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return corrupted


def corrupt_jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [90, 70, 50, 30, 10][severity - 1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, 1)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def corrupt_low_light(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [0.8, 0.6, 0.4, 0.2, 0.1][severity - 1]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def corrupt_rotation(img: np.ndarray, severity: int) -> np.ndarray:
    angle = [10, 20, 30, 45, 90][severity - 1]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def corrupt_occlusion(img: np.ndarray, severity: int) -> np.ndarray:
    pct = [0.05, 0.10, 0.20, 0.35, 0.50][severity - 1]
    h, w = img.shape[:2]
    occ_h, occ_w = int(h * np.sqrt(pct)), int(w * np.sqrt(pct))
    y = np.random.randint(0, max(1, h - occ_h))
    x = np.random.randint(0, max(1, w - occ_w))
    corrupted = img.copy()
    corrupted[y:y+occ_h, x:x+occ_w] = 0
    return corrupted


CORRUPTIONS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "blur": corrupt_blur,
    "noise": corrupt_noise,
    "jpeg_compression": corrupt_jpeg,
    "low_light": corrupt_low_light,
    "rotation": corrupt_rotation,
    "occlusion": corrupt_occlusion,
}


class CorruptedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, corruption_fn, severity: int, transform):
        self.df = df.reset_index(drop=True)
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.corruption_fn(img, self.severity)
        augmented = self.transform(image=img)
        return {
            "image": augmented["image"],
            "label": int(row["label"]),
        }


@torch.no_grad()
def evaluate_robustness_level(
    model: nn.Module,
    df: pd.DataFrame,
    corruption_name: str,
    severity: int,
    transform,
    device: torch.device,
) -> Dict[str, float]:
    fn = CORRUPTIONS[corruption_name]
    dataset = CorruptedDataset(df, fn, severity, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["label"].numpy()
        logits = model(images)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

    return {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
    }


def run_all_robustness_tests(
    model: nn.Module,
    df: pd.DataFrame,
    transform,
    device: torch.device,
) -> pd.DataFrame:
    records = []
    for c_name in CORRUPTIONS.keys():
        for sev in range(1, 6):
            res = evaluate_robustness_level(model, df, c_name, sev, transform, device)
            records.append({
                "corruption": c_name,
                "severity": sev,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
            })
    return pd.DataFrame(records)


def plot_robustness_curves(df: pd.DataFrame, output_path: Path, baseline_f1: float = 0.96) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for c_name, grp in df.groupby("corruption"):
        ax.plot(grp["severity"], grp["macro_f1"], marker="o", label=c_name)

    ax.axhline(baseline_f1, color="black", linestyle="--", alpha=0.7, label=f"Clean Test ({baseline_f1:.2f})")
    ax.set_xlabel("Severity Level (1 to 5)", fontsize=12)
    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.set_title("Model Robustness Under Environmental Corruptions", fontsize=14)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_ood_rejection(
    model: nn.Module,
    test_loader: DataLoader,
    ood_threshold: float = 0.60,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Max-Softmax OOD Uncertainty Rejection.
    Predictions with max softmax probability < ood_threshold are flagged as 'Uncertain / Non-leaf'.
    """
    model.eval()
    max_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            probs = torch.softmax(model(images), dim=-1)
            max_p, _ = torch.max(probs, dim=-1)
            max_probs.extend(max_p.cpu().numpy())

    max_probs = np.array(max_probs)
    accepted_fraction = float(np.mean(max_probs >= ood_threshold))
    rejected_fraction = 1.0 - accepted_fraction

    return {
        "threshold": ood_threshold,
        "accepted_rate": accepted_fraction,
        "rejected_rate": rejected_fraction,
        "mean_confidence": float(np.mean(max_probs)),
    }
