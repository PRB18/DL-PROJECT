"""
tests/test_data.py
------------------
Unit tests for the data pipeline (Phase 1).
Run with: pytest tests/test_data.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image_dir(tmp_path: Path) -> Path:
    """Create a minimal dataset directory tree with fake images."""
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Pepper___Bacterial_spot"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(10):
            # Create a valid 256×256 RGB PNG
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"img_{i:04d}.png")
    return tmp_path


@pytest.fixture
def dummy_manifest(tmp_path: Path, dummy_image_dir: Path) -> Path:
    """Create a minimal manifest CSV from the dummy image directory."""
    records = []
    for cls_dir in dummy_image_dir.iterdir():
        for img_path in cls_dir.iterdir():
            records.append({
                "path": str(img_path),
                "class_name": cls_dir.name,
                "corrupted": False,
                "duplicate_group": hash(img_path.stem) % 100,
                "width": 256,
                "height": 256,
                "phash": "abc123",
            })
    df = pd.DataFrame(records)
    out = tmp_path / "manifest.csv"
    df.to_csv(out, index=False)
    return out


# ──────────────────────────────────────────────────────────────
# Tests: EDA helpers
# ──────────────────────────────────────────────────────────────

class TestEDA:
    def test_collect_image_paths(self, dummy_image_dir: Path):
        from src.data.eda import collect_image_paths
        df = collect_image_paths(dummy_image_dir)
        assert len(df) == 30, f"Expected 30 images, got {len(df)}"
        assert "path" in df.columns
        assert "class_name" in df.columns
        assert df["class_name"].nunique() == 3

    def test_safe_load_valid_image(self, tmp_path: Path):
        from src.data.eda import _safe_load
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        p = tmp_path / "test.png"
        img.save(p)
        result = _safe_load(p)
        assert result is not None

    def test_safe_load_corrupted_file(self, tmp_path: Path):
        from src.data.eda import _safe_load
        p = tmp_path / "bad.png"
        p.write_bytes(b"this is not a valid image file")
        result = _safe_load(p)
        assert result is None


# ──────────────────────────────────────────────────────────────
# Tests: Splits
# ──────────────────────────────────────────────────────────────

class TestSplits:
    def test_no_leakage(self, dummy_manifest: Path, tmp_path: Path):
        from src.data.splits import build_splits, _validate_no_leakage
        splits = build_splits(
            manifest_csv=dummy_manifest,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            output_dir=tmp_path / "splits",
        )
        # Should not raise
        _validate_no_leakage(splits["train"], splits["val"], splits["test"])

    def test_split_sizes(self, dummy_manifest: Path, tmp_path: Path):
        from src.data.splits import build_splits
        splits = build_splits(
            manifest_csv=dummy_manifest,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            output_dir=tmp_path / "splits",
        )
        total = sum(len(v) for v in splits.values())
        # Total after dedup should equal unique items in manifest
        assert total > 0
        # Train should be the largest split
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_class_map_saved(self, dummy_manifest: Path, tmp_path: Path):
        from src.data.splits import build_splits
        splits_dir = tmp_path / "splits"
        build_splits(
            manifest_csv=dummy_manifest,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            output_dir=splits_dir,
        )
        class_map = splits_dir / "plantvillage_class_map.json"
        assert class_map.exists()
        with open(class_map) as f:
            data = json.load(f)
        assert "class_to_idx" in data
        assert "idx_to_class" in data


# ──────────────────────────────────────────────────────────────
# Tests: Dataset
# ──────────────────────────────────────────────────────────────

class TestDataset:
    def _make_df(self, dummy_image_dir: Path) -> pd.DataFrame:
        records = []
        for i, cls_dir in enumerate(dummy_image_dir.iterdir()):
            for img_path in cls_dir.iterdir():
                records.append({
                    "path": str(img_path),
                    "class_name": cls_dir.name,
                    "label": i,
                })
        return pd.DataFrame(records)

    def test_dataset_len(self, dummy_image_dir: Path):
        from src.data.dataset import PlantDiseaseDataset, get_val_transforms
        df = self._make_df(dummy_image_dir)
        ds = PlantDiseaseDataset(df, transform=get_val_transforms(64))
        assert len(ds) == len(df)

    def test_dataset_item_shape(self, dummy_image_dir: Path):
        from src.data.dataset import PlantDiseaseDataset, get_val_transforms
        df = self._make_df(dummy_image_dir)
        ds = PlantDiseaseDataset(df, transform=get_val_transforms(64))
        item = ds[0]
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 64, 64)
        assert isinstance(item["label"], int)

    def test_train_transforms_applied(self, dummy_image_dir: Path):
        from src.data.dataset import PlantDiseaseDataset, get_train_transforms
        df = self._make_df(dummy_image_dir)
        ds = PlantDiseaseDataset(df, transform=get_train_transforms(64))
        # With augmentation, images should still be valid tensors
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)
        assert not torch.isnan(item["image"]).any()

    def test_background_randomisation(self, dummy_image_dir: Path):
        from src.data.dataset import PlantDiseaseDataset, get_train_transforms
        df = self._make_df(dummy_image_dir)
        ds = PlantDiseaseDataset(df, transform=get_train_transforms(64),
                                  background_rand=True)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)


# ──────────────────────────────────────────────────────────────
# Tests: Class weights
# ──────────────────────────────────────────────────────────────

class TestClassWeights:
    def test_shape(self):
        from src.data.class_weights import compute_class_weights
        df = pd.DataFrame({"label": [0]*100 + [1]*50 + [2]*10})
        w = compute_class_weights(df, num_classes=3)
        assert w.shape == (3,)

    def test_minority_gets_higher_weight(self):
        from src.data.class_weights import compute_class_weights
        df = pd.DataFrame({"label": [0]*100 + [1]*10})
        w = compute_class_weights(df, num_classes=2)
        # Class 1 (minority) should have higher weight
        assert w[1] > w[0], f"Expected w[1] > w[0], got {w}"

    def test_all_methods_run(self):
        from src.data.class_weights import compute_class_weights
        df = pd.DataFrame({"label": [0]*100 + [1]*50 + [2]*20})
        for method in ["inverse_freq", "sqrt_inverse", "effective_num"]:
            w = compute_class_weights(df, num_classes=3, method=method)
            assert w.shape == (3,)
            assert (w > 0).all()


# ──────────────────────────────────────────────────────────────
# Tests: Seed utility
# ──────────────────────────────────────────────────────────────

class TestSeed:
    def test_reproducibility(self):
        from src.utils.seed import set_global_seed
        set_global_seed(42)
        a = torch.randn(5)
        set_global_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b), "Seeds not producing identical tensors"

    def test_different_seeds(self):
        from src.utils.seed import set_global_seed
        set_global_seed(42)
        a = torch.randn(5)
        set_global_seed(99)
        b = torch.randn(5)
        assert not torch.allclose(a, b), "Different seeds should give different tensors"
