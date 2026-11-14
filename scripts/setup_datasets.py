"""
scripts/setup_datasets.py
Helper utility to initialize dataset directories and copy local sample images
for immediate zero-shot evaluation and development.
"""

import shutil
from pathlib import Path


def setup_directories():
    project_root = Path(__file__).resolve().parent.parent

    dirs = [
        project_root / "datasets" / "whiteboard",
        project_root / "datasets" / "handwriting",
        project_root / "datasets" / "campus_test_set",
        project_root / "runs" / "whiteboard_optimization",
        project_root / "runs" / "campus_eval"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Verified directory: {d}")

    # Copy local sample test images to campus_test_set
    campus_dir = project_root / "datasets" / "campus_test_set"
    samples = ["board-srmap.jpg", "test_user_wb.jpg", "user_test_board.jpg"]

    copied = 0
    for s in samples:
        src = project_root / s
        if src.exists():
            dst = campus_dir / s
            shutil.copy(src, dst)
            copied += 1

    print(f"\nCopied {copied} sample classroom photos into datasets/campus_test_set/")
    print("\nDataset Instructions:")
    print("1. Peshawar Whiteboard Dataset (YOLOv8 format):")
    print("   Place train/, valid/, test/ and data.yaml under datasets/whiteboard/")
    print("2. IAM Handwriting Parquet Splits:")
    print("   Place train.parquet, validation.parquet, test.parquet under datasets/handwriting/")
    print("3. Campus Zero-Shot Test Set:")
    print("   Add your 15-20 college classroom photos to datasets/campus_test_set/")
    print("\nSetup completed successfully!")


if __name__ == "__main__":
    setup_directories()
