"""
scripts/download_datasets.py
Automated Dataset Downloader and Setup Script for BoardMind AI.

Downloads and organizes:
1. IAM Handwriting Line Dataset (Teklia/IAM-line from Hugging Face: 10,373 line samples)
   - Splits: Train (6,482), Validation (976), Test (2,915)
   - Destination: datasets/handwriting/
2. University of Peshawar Whiteboard Dataset (Roboflow Universe: 1,964 annotated images)
   - Splits: Train (1,608), Valid (179), Test (177)
   - Destination: datasets/whiteboard/
3. Campus Classroom Test Set (SRM AP campus photos)
   - Destination: datasets/campus_test_set/
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_directories():
    """Ensure all dataset destination directories exist."""
    dirs = [
        PROJECT_ROOT / "datasets" / "handwriting",
        PROJECT_ROOT / "datasets" / "whiteboard",
        PROJECT_ROOT / "datasets" / "campus_test_set",
        PROJECT_ROOT / "runs" / "whiteboard_optimization",
        PROJECT_ROOT / "runs" / "campus_eval"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[1/3] Dataset directories verified.")


def download_iam_handwriting():
    """
    Downloads the Teklia/IAM-line Parquet dataset files from Hugging Face
    matching the exact splits from notebooks/01_dataset_exploration.ipynb:
    Train: 6,482 | Validation: 976 | Test: 2,915 (Total: 10,373 lines).
    """
    target_dir = PROJECT_ROOT / "datasets" / "handwriting"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("\n[2/3] Checking IAM Handwriting Dataset (Hugging Face Teklia/IAM-line)...")

    # Method A: Try using Hugging Face datasets library if installed
    try:
        from datasets import load_dataset
        print("  Loading Teklia/IAM-line via Hugging Face datasets library...")
        ds = load_dataset("Teklia/IAM-line")
        ds["train"].to_parquet(target_dir / "train.parquet")
        ds["validation"].to_parquet(target_dir / "validation.parquet")
        ds["test"].to_parquet(target_dir / "test.parquet")
        print("  ✓ All 3 Parquet files successfully saved to datasets/handwriting/ via datasets library.")
        return
    except Exception as e:
        print(f"  (datasets library not available or failed: {e}. Falling back to direct URL download.)")

    # Method B: Direct URL download from Hugging Face
    files = {
        "train.parquet": "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/train.parquet",
        "validation.parquet": "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/validation.parquet",
        "test.parquet": "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/test.parquet"
    }

    for filename, url in files.items():
        dest = target_dir / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  ✓ {filename} already exists ({dest.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
            continue

        print(f"  ⬇ Downloading {filename} from Hugging Face...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, str(dest))
            print(f"  ✓ Saved to {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  ⚠ Could not auto-download {filename}: {e}")
            print(f"    You can load it via Python: datasets.load_dataset('Teklia/IAM-line')")


def setup_campus_test_set():
    """Copies root test images into campus_test_set."""
    print("\n[3/3] Setting up Campus Zero-Shot Test Set...")
    campus_dir = PROJECT_ROOT / "datasets" / "campus_test_set"

    sample_sources = [
        PROJECT_ROOT / "test_user_wb.jpg",
        PROJECT_ROOT / "user_test_board.jpg",
        PROJECT_ROOT / "notebooks" / "board-srmap.jpg"
    ]

    copied = 0
    for src in sample_sources:
        if src.exists():
            dst = campus_dir / src.name
            if not dst.exists():
                shutil.copy(src, dst)
                copied += 1

    print(f"  ✓ Campus test set ready: {len(list(campus_dir.glob('*.jpg')))} images in {campus_dir}")


def print_roboflow_instructions():
    """Instructions for downloading the 1,964-image Whiteboard dataset."""
    wb_dir = PROJECT_ROOT / "datasets" / "whiteboard"
    yaml_file = wb_dir / "data.yaml"

    print("\n" + "=" * 75)
    print("WHITEBOARD DETECTION DATASET STATUS:")
    print("=" * 75)
    if yaml_file.exists():
        print(f"✓ Whiteboard dataset found at {wb_dir}")
    else:
        print(f"⚠ Whiteboard dataset folder ({wb_dir}) does not contain data.yaml.")
        print("\nTo download the University of Peshawar Whiteboard Dataset (1,964 images):")
        print("Option A (Roboflow Python API):")
        print("  pip install roboflow")
        print("  python -c \"from roboflow import Roboflow; rf=Roboflow(); rf.workspace('university-of-peshawar-4qet5').project('whiteboard-dataset').version(1).download('yolov8', location='datasets/whiteboard')\"")
        print("\nOption B (Manual Web Download):")
        print("  1. Visit: https://universe.roboflow.com/university-of-peshawar-4qet5/whiteboard-dataset")
        print("  2. Click 'Download Dataset' -> Format: 'YOLOv8' -> Download zip")
        print("  3. Extract contents into: datasets/whiteboard/")
    print("=" * 75)


if __name__ == "__main__":
    setup_directories()
    download_iam_handwriting()
    setup_campus_test_set()
    print_roboflow_instructions()
