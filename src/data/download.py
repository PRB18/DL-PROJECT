"""
src/data/download.py
--------------------
Automated dataset download helpers.

Datasets used
─────────────
1. PlantVillage  — lab-style images, clean backgrounds, 38 classes, ~54 k images.
   Source: https://www.kaggle.com/datasets/emmarex/plantdisease
   Alternative: torchvision does NOT host PlantVillage, so we use the Kaggle API.

2. PlantDoc      — in-the-wild images (field photos, cluttered backgrounds).
   Source: https://github.com/pratikkayal/PlantDoc-Dataset
   We use this ONLY as an out-of-distribution / domain-shift test set.

WHY TWO DATASETS?
─────────────────
PlantVillage images are photographed against a plain white/grey background in
controlled lab lighting. A model can reach 99 % accuracy by learning background
colour instead of leaf lesion patterns. When you deploy that model on a real
farm photo it will likely fail. PlantDoc provides real-world photos with
cluttered, natural backgrounds — exactly the distribution a farmer's phone
would produce. By reporting the accuracy gap (PlantVillage test vs PlantDoc),
we give an honest, production-relevant benchmark.
"""

import os
import subprocess
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Kaggle download (requires ~/.kaggle/kaggle.json)
# ──────────────────────────────────────────────────────────────

def download_plantvillage_kaggle(dest: Path = Path("data/raw")) -> Path:
    """
    Download PlantVillage via the Kaggle CLI.

    Prerequisites
    -------------
    1. pip install kaggle
    2. Place kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME / KAGGLE_KEY env vars)
    3. Accept the dataset's terms on the Kaggle website first.

    Returns
    -------
    Path to the unzipped PlantVillage directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    pv_dir = dest / "PlantVillage"

    if pv_dir.exists():
        print(f"[skip] PlantVillage already found at {pv_dir}")
        return pv_dir

    print("Downloading PlantVillage from Kaggle …")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "emmarex/plantdisease",
        "-p", str(dest),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    # The Kaggle unzip usually creates a 'PlantVillage' subdirectory
    return pv_dir


# ──────────────────────────────────────────────────────────────
# PlantDoc (GitHub release)
# ──────────────────────────────────────────────────────────────

PLANTDOC_URL = (
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
)


def download_plantdoc(dest: Path = Path("data/raw")) -> Path:
    """
    Download and unzip PlantDoc from GitHub.

    Returns
    -------
    Path to the unzipped PlantDoc directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    pd_dir = dest / "PlantDoc"

    if pd_dir.exists():
        print(f"[skip] PlantDoc already found at {pd_dir}")
        return pd_dir

    zip_path = dest / "PlantDoc.zip"
    print("Downloading PlantDoc …")
    _download_file(PLANTDOC_URL, zip_path)

    print("Unzipping PlantDoc …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Rename the GitHub archive folder to 'PlantDoc'
    extracted = dest / "PlantDoc-Dataset-master"
    if extracted.exists():
        extracted.rename(pd_dir)

    zip_path.unlink(missing_ok=True)
    return pd_dir


def _download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Stream-download a URL to disk with a progress bar."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    parser.add_argument("--plantvillage", action="store_true", default=True)
    parser.add_argument("--plantdoc",     action="store_true", default=True)
    args = parser.parse_args()

    dest = Path(args.dest)

    if args.plantvillage:
        pv = download_plantvillage_kaggle(dest)
        print(f"PlantVillage → {pv}")

    if args.plantdoc:
        pd = download_plantdoc(dest)
        print(f"PlantDoc     → {pd}")
