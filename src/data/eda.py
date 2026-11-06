"""
src/data/eda.py
---------------
Exploratory Data Analysis for PlantVillage (and optionally PlantDoc).

What this script does
─────────────────────
1. Counts images per class → class distribution bar chart.
2. Samples a grid of images per class → visual sanity check.
3. Measures image sizes (width × height) → distribution plots.
4. Detects near-duplicate images using perceptual hashing (imagehash).
5. Identifies corrupted/unreadable files.
6. Computes and plots class imbalance ratio (max/min class size).
7. Saves all figures to reports/figures/.

Design decisions
────────────────
• We use imagehash (average hash) with a Hamming-distance threshold of ≤ 8 bits
  to flag near-duplicates. This catches re-compressed JPEGs and slight crops
  of the same underlying photo — the most common form of unintentional leakage.
• We log everything to a CSV so the split builder can avoid putting duplicates
  in both train and test.
• We separate PlantVillage EDA from PlantDoc EDA because the distributions are
  different and comparing them is itself informative.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_load(path: Path) -> Optional[Image.Image]:
    """Load an image; return None and log if corrupted."""
    try:
        img = Image.open(path)
        img.verify()          # checks file integrity without decoding pixels
        img = Image.open(path)  # re-open after verify() (file pointer reset)
        img.load()
        return img
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


def collect_image_paths(root: Path) -> pd.DataFrame:
    """
    Walk a dataset directory structured as root/<class_name>/<image>.

    Returns a DataFrame with columns: [path, class_name, split].
    """
    records = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                records.append({"path": str(img_path), "class_name": cls_dir.name})
    df = pd.DataFrame(records)
    print(f"Found {len(df):,} images in {root}")
    return df


# ──────────────────────────────────────────────────────────────
# 1. Class distribution
# ──────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, dataset_name: str) -> None:
    counts = df["class_name"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.28)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of images")
    ax.set_title(f"{dataset_name} — Class Distribution ({len(counts)} classes)")

    # Annotate imbalance ratio
    ratio = counts.max() / counts.min()
    ax.text(0.98, 0.02, f"Imbalance ratio: {ratio:.1f}×",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    out = FIGURE_DIR / f"{dataset_name.lower()}_class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────
# 2. Sample grid
# ──────────────────────────────────────────────────────────────

def plot_sample_grid(df: pd.DataFrame, dataset_name: str,
                     n_classes: int = 10, n_per_class: int = 5) -> None:
    classes = df["class_name"].unique()[:n_classes]
    fig = plt.figure(figsize=(n_per_class * 2.5, n_classes * 2.5))
    gs = gridspec.GridSpec(n_classes, n_per_class, figure=fig,
                           wspace=0.05, hspace=0.4)

    for row, cls in enumerate(classes):
        samples = df[df["class_name"] == cls]["path"].sample(
            min(n_per_class, len(df[df["class_name"] == cls])),
            random_state=42
        )
        for col, path in enumerate(samples):
            ax = fig.add_subplot(gs[row, col])
            img = _safe_load(Path(path))
            if img is not None:
                ax.imshow(img.convert("RGB"))
            ax.axis("off")
            if col == 0:
                # Shorten long class names for readability
                label = cls.replace("___", "\n").replace("_", " ")
                ax.set_ylabel(label, fontsize=7, rotation=0,
                              ha="right", va="center")

    fig.suptitle(f"{dataset_name} — Sample Grid", fontsize=14)
    out = FIGURE_DIR / f"{dataset_name.lower()}_sample_grid.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ──────────────────────────────────────────────────────────────
# 3. Image size distribution
# ──────────────────────────────────────────────────────────────

def analyze_image_sizes(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    widths, heights, areas = [], [], []
    corrupted = []

    for path in tqdm(df["path"], desc=f"[{dataset_name}] Measuring sizes"):
        img = _safe_load(Path(path))
        if img is None:
            corrupted.append(path)
            widths.append(0); heights.append(0); areas.append(0)
            continue
        w, h = img.size
        widths.append(w); heights.append(h); areas.append(w * h)

    df = df.copy()
    df["width"] = widths
    df["height"] = heights
    df["area"] = areas
    df["corrupted"] = df["path"].isin(corrupted)

    if corrupted:
        print(f"⚠️  {len(corrupted)} corrupted files detected in {dataset_name}:")
        for c in corrupted[:10]:
            print(f"   {c}")
        if len(corrupted) > 10:
            print(f"   … and {len(corrupted) - 10} more.")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    valid = df[~df["corrupted"]]

    axes[0].hist(valid["width"],  bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Width distribution"); axes[0].set_xlabel("pixels")

    axes[1].hist(valid["height"], bins=40, color="coral",    edgecolor="white")
    axes[1].set_title("Height distribution"); axes[1].set_xlabel("pixels")

    axes[2].scatter(valid["width"], valid["height"], alpha=0.3,
                    s=5, c="seagreen")
    axes[2].set_title("Width × Height scatter")
    axes[2].set_xlabel("width"); axes[2].set_ylabel("height")

    fig.suptitle(f"{dataset_name} — Image Size Analysis")
    plt.tight_layout()
    out = FIGURE_DIR / f"{dataset_name.lower()}_image_sizes.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return df


# ──────────────────────────────────────────────────────────────
# 4. Duplicate detection (perceptual hashing)
# ──────────────────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    """
    Compute average perceptual hash (aHash) for each image.
    Flag pairs with Hamming distance ≤ threshold as near-duplicates.

    Returns the df with a 'phash' column and a 'duplicate_group' column.

    Design rationale
    ─────────────────
    aHash is fast (< 1 ms per image) and reliably catches:
      - Re-compressed JPEGs of the same photo
      - Slight crops / resize artefacts
      - Colour-adjusted copies (at high threshold)
    threshold=8 on a 64-bit hash means ≤ 12.5% bit difference — empirically
    good for catching copies while not falsely flagging distinct leaves.
    """
    hashes = []
    for path in tqdm(df["path"], desc="Computing perceptual hashes"):
        img = _safe_load(Path(path))
        h = str(imagehash.average_hash(img)) if img is not None else "corrupted"
        hashes.append(h)

    df = df.copy()
    df["phash"] = hashes

    # Group near-duplicates using a simple greedy merge
    from collections import defaultdict
    groups: dict[str, int] = {}
    group_id = 0

    unique_hashes = df[df["phash"] != "corrupted"]["phash"].unique()
    # Convert to imagehash objects for Hamming comparison
    hash_objs = {h: imagehash.hex_to_hash(h) for h in unique_hashes}

    # Build adjacency list of near-duplicate pairs (O(n²) but n is manageable)
    adjacency: dict[str, set] = defaultdict(set)
    hash_list = list(unique_hashes)
    for i, h1 in enumerate(hash_list):
        for h2 in hash_list[i+1:]:
            if hash_objs[h1] - hash_objs[h2] <= threshold:
                adjacency[h1].add(h2)
                adjacency[h2].add(h1)

    # BFS to assign group IDs to connected components
    visited = set()
    for h in hash_list:
        if h not in visited:
            # BFS
            queue = [h]
            component = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adjacency[node] - visited)
            for node in component:
                groups[node] = group_id
            group_id += 1

    df["duplicate_group"] = df["phash"].map(lambda h: groups.get(h, -1))

    # Count how many groups have > 1 member
    group_sizes = df.groupby("duplicate_group").size()
    dup_groups = group_sizes[group_sizes > 1]
    n_dup_images = int((group_sizes[group_sizes > 1] - 1).sum())
    print(f"Near-duplicate groups: {len(dup_groups)} | "
          f"Redundant images: {n_dup_images} ({100*n_dup_images/len(df):.1f}%)")

    return df


# ──────────────────────────────────────────────────────────────
# 5. Main EDA runner
# ──────────────────────────────────────────────────────────────

def run_eda(data_root: Path, dataset_name: str,
            check_duplicates: bool = True) -> pd.DataFrame:
    """Full EDA pipeline. Returns annotated DataFrame."""
    print(f"\n{'='*60}")
    print(f"  EDA: {dataset_name}")
    print(f"{'='*60}")

    df = collect_image_paths(data_root)

    if df.empty:
        print(f"⚠️  No images found under {data_root}. "
              f"Did you download the dataset? Run: python src/data/download.py")
        return df

    plot_class_distribution(df, dataset_name)
    plot_sample_grid(df, dataset_name)
    df = analyze_image_sizes(df, dataset_name)

    if check_duplicates:
        df = detect_duplicates(df)

    # Save annotated manifest
    out_csv = FIGURE_DIR.parent / f"{dataset_name.lower()}_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"Manifest saved → {out_csv}")

    # Summary stats
    print("\n--- Summary ---")
    print(f"Total images   : {len(df):,}")
    print(f"Classes        : {df['class_name'].nunique()}")
    print(f"Corrupted      : {df['corrupted'].sum()}")
    if check_duplicates and "duplicate_group" in df.columns:
        dup_groups = df.groupby("duplicate_group").size()
        n_dup = int((dup_groups[dup_groups > 1] - 1).sum())
        print(f"Near-duplicates: {n_dup}")
    print(f"Width range    : {df['width'].min()}–{df['width'].max()} px")
    print(f"Height range   : {df['height'].min()}–{df['height'].max()} px")

    return df


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA on plant disease datasets")
    parser.add_argument("--plantvillage", default="data/raw/PlantVillage",
                        help="Path to PlantVillage root")
    parser.add_argument("--plantdoc",     default="data/raw/PlantDoc",
                        help="Path to PlantDoc root")
    parser.add_argument("--no-duplicates", action="store_true",
                        help="Skip duplicate detection (faster)")
    args = parser.parse_args()

    pv_df = run_eda(Path(args.plantvillage), "PlantVillage",
                    check_duplicates=not args.no_duplicates)

    pd_path = Path(args.plantdoc)
    if pd_path.exists():
        pd_df = run_eda(pd_path, "PlantDoc",
                        check_duplicates=not args.no_duplicates)
    else:
        print(f"\n[skip] PlantDoc not found at {pd_path}. "
              f"Download with: python src/data/download.py --plantdoc")
