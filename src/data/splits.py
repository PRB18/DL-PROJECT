"""
src/data/splits.py
------------------
Build leak-free, stratified train/val/test splits.

The splitting algorithm
────────────────────────
1. Load the EDA manifest (which has duplicate_group labels).
2. Remove corrupted images.
3. For near-duplicates: keep ONE representative per group, discard the rest.
   This prevents the same photo from appearing in both train and test.
4. Stratified split (sklearn StratifiedShuffleSplit) to maintain class balance
   across all three folds.
5. Save three CSV files: train.csv, val.csv, test.csv.
6. Save a separate plantdoc_test.csv (domain-shift evaluation set).

Assumption / risk flagged
─────────────────────────
PlantVillage has ~38 classes; some have as few as 152 images. After
deduplication and 15 % test split, the smallest class will have ~129 test
images — enough for a meaningful per-class F1 but on the edge. We log this.
If you have fewer than 50 test images per class, treat per-class metrics with
caution (wide confidence intervals).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ──────────────────────────────────────────────────────────────
# Core split builder
# ──────────────────────────────────────────────────────────────

def build_splits(
    manifest_csv: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    output_dir: Path = Path("data/splits"),
    dataset_prefix: str = "plantvillage",
) -> dict[str, pd.DataFrame]:
    """
    Build leak-free stratified splits from an EDA manifest.

    Parameters
    ----------
    manifest_csv  : Path to the CSV produced by eda.run_eda().
    val_frac      : Fraction of data for validation.
    test_frac     : Fraction of data for test.
    seed          : Random seed (must match config.yaml).
    output_dir    : Where to save train/val/test CSVs.
    dataset_prefix: Prefix for output filenames.

    Returns
    -------
    Dict with keys 'train', 'val', 'test'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    print(f"Loaded manifest: {len(df):,} images, {df['class_name'].nunique()} classes")

    # ── Step 1: Remove corrupted images ──────────────────────
    before = len(df)
    df = df[~df["corrupted"]].copy()
    print(f"Removed {before - len(df)} corrupted images")

    # ── Step 2: Deduplicate ──────────────────────────────────
    if "duplicate_group" in df.columns:
        # Keep only the first image in each duplicate group
        df = df.sort_values("path")  # deterministic order
        df = df.drop_duplicates(subset="duplicate_group", keep="first")
        print(f"After deduplication: {len(df):,} images")
    else:
        print("⚠️  No duplicate_group column found. Skipping deduplication.")

    # ── Step 3: Encode class labels ───────────────────────────
    class_names = sorted(df["class_name"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    df["label"] = df["class_name"].map(class_to_idx)

    # ── Step 4: Stratified split ──────────────────────────────
    # First split off test set
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    train_val_idx, test_idx = next(sss_test.split(df, df["label"]))
    df_train_val = df.iloc[train_val_idx].copy()
    df_test      = df.iloc[test_idx].copy()

    # Then split train_val into train and val
    # val_frac relative to the full dataset → adjust for train_val subset
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_adjusted, random_state=seed
    )
    train_idx, val_idx = next(sss_val.split(df_train_val, df_train_val["label"]))
    df_train = df_train_val.iloc[train_idx].copy()
    df_val   = df_train_val.iloc[val_idx].copy()

    # ── Step 5: Validation checks ─────────────────────────────
    _validate_no_leakage(df_train, df_val, df_test)
    _report_split_stats(df_train, df_val, df_test, class_names)

    # ── Step 6: Save ─────────────────────────────────────────
    splits = {"train": df_train, "val": df_val, "test": df_test}
    for name, split_df in splits.items():
        out = output_dir / f"{dataset_prefix}_{name}.csv"
        split_df.to_csv(out, index=False)
        print(f"Saved {out}  ({len(split_df):,} images)")

    # Save class map
    class_map_path = output_dir / f"{dataset_prefix}_class_map.json"
    with open(class_map_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx,
                   "idx_to_class": {str(v): k for k, v in class_to_idx.items()}}, f,
                  indent=2)
    print(f"Class map saved → {class_map_path}")

    return splits


def _validate_no_leakage(train: pd.DataFrame,
                          val: pd.DataFrame,
                          test: pd.DataFrame) -> None:
    """Assert zero path overlap between splits."""
    t_paths = set(train["path"])
    v_paths = set(val["path"])
    te_paths = set(test["path"])

    tv_overlap = t_paths & v_paths
    tt_overlap = t_paths & te_paths
    vt_overlap = v_paths & te_paths

    assert len(tv_overlap) == 0, f"DATA LEAKAGE: {len(tv_overlap)} paths in train ∩ val"
    assert len(tt_overlap) == 0, f"DATA LEAKAGE: {len(tt_overlap)} paths in train ∩ test"
    assert len(vt_overlap) == 0, f"DATA LEAKAGE: {len(vt_overlap)} paths in val ∩ test"
    print("✅  No data leakage detected across splits.")


def _report_split_stats(train: pd.DataFrame,
                         val: pd.DataFrame,
                         test: pd.DataFrame,
                         class_names: list[str]) -> None:
    """Print per-split counts and warn about small test classes."""
    total = len(train) + len(val) + len(test)
    print(f"\n{'Split':10s} {'Images':>8s} {'%':>6s}")
    print("-" * 28)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:10s} {len(df):>8,} {100*len(df)/total:>5.1f}%")

    # Per-class test counts
    test_counts = test["class_name"].value_counts()
    small_classes = test_counts[test_counts < 50]
    if not small_classes.empty:
        print(f"\n⚠️  {len(small_classes)} classes have < 50 test images. "
              "Per-class F1 for these classes has wide confidence intervals:")
        print(small_classes.to_string())


# ──────────────────────────────────────────────────────────────
# PlantDoc domain-shift split
# ──────────────────────────────────────────────────────────────

def build_plantdoc_split(
    plantdoc_manifest: Path,
    plantvillage_class_map: Path,
    output_dir: Path = Path("data/splits"),
) -> Optional[pd.DataFrame]:
    """
    Build a PlantDoc test split restricted to classes that overlap with
    PlantVillage (by fuzzy name matching).

    Returns DataFrame or None if PlantDoc was not found.
    """
    if not plantdoc_manifest.exists():
        print(f"⚠️  PlantDoc manifest not found at {plantdoc_manifest}. "
              "Run eda.py first.")
        return None

    with open(plantvillage_class_map) as f:
        pv_map = json.load(f)
    pv_classes = set(pv_map["class_to_idx"].keys())

    pd_df = pd.read_csv(plantdoc_manifest)
    pd_df = pd_df[~pd_df["corrupted"]]

    # Fuzzy match: PlantDoc uses different naming conventions than PlantVillage.
    # We try lowercase partial matching. Manual review is recommended.
    def _map_plantdoc_class(cls: str) -> Optional[str]:
        cls_lower = cls.lower().replace(" ", "_").replace("-", "_")
        for pv_cls in pv_classes:
            pv_lower = pv_cls.lower().replace(" ", "_").replace("-", "_")
            # Match if the crop and disease keywords overlap
            if cls_lower in pv_lower or pv_lower in cls_lower:
                return pv_cls
        return None

    pd_df = pd_df.copy()
    pd_df["pv_class"] = pd_df["class_name"].map(_map_plantdoc_class)
    matched = pd_df[pd_df["pv_class"].notna()].copy()

    print(f"PlantDoc: {len(pd_df)} images, {pd_df['class_name'].nunique()} classes")
    print(f"Matched to PlantVillage: {len(matched)} images, "
          f"{matched['class_name'].nunique()} PlantDoc classes → "
          f"{matched['pv_class'].nunique()} PV classes")

    if matched.empty:
        print("⚠️  No overlapping classes found. Check class names manually.")
        return None

    out = output_dir / "plantdoc_test.csv"
    matched.to_csv(out, index=False)
    print(f"PlantDoc test split saved → {out}")
    return matched


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.utils.seed import set_global_seed

    parser = argparse.ArgumentParser(description="Build data splits")
    parser.add_argument("--manifest",    default="reports/plantvillage_manifest.csv")
    parser.add_argument("--pd-manifest", default="reports/plantdoc_manifest.csv")
    parser.add_argument("--output",      default="data/splits")
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    splits = build_splits(
        manifest_csv=Path(args.manifest),
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
        output_dir=Path(args.output),
    )

    class_map = Path(args.output) / "plantvillage_class_map.json"
    if class_map.exists() and Path(args.pd_manifest).exists():
        build_plantdoc_split(
            plantdoc_manifest=Path(args.pd_manifest),
            plantvillage_class_map=class_map,
            output_dir=Path(args.output),
        )
