"""
src/evaluate_campus_set.py
Zero-Shot Domain Adaptation Benchmark for Real Classroom Whiteboards.

Evaluates the champion YOLOv8n detector and homography rectification pipeline
on unconstrained classroom photos (angled views, whiteboard borders, ambient glare)
collected from university lecture halls.

Fulfills the Custom Classroom Whiteboard Evaluation objective.
"""

import sys
import time
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.processing.image_preprocessor import get_image_preprocessor


def run_campus_evaluation():
    project_root = Path(__file__).resolve().parent.parent
    detector = get_whiteboard_detector()
    preprocessor = get_image_preprocessor()

    # Search for campus images in datasets/campus_test_set or root test images
    campus_dir = project_root / "datasets" / "campus_test_set"
    image_files = []
    if campus_dir.exists():
        image_files = list(campus_dir.glob("*.jpg")) + list(campus_dir.glob("*.png"))

    if not image_files:
        # Fallback to root sample photos
        for name in ["board-srmap.jpg", "test_user_wb.jpg", "user_test_board.jpg"]:
            p = project_root / name
            if p.exists():
                image_files.append(p)

    out_dir = project_root / "runs" / "campus_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BOARDMIND AI: REAL CLASSROOM ZERO-SHOT DOMAIN BENCHMARK")
    print(f"Target Images Found: {len(image_files)}")
    print(f"Output Directory:    {out_dir}")
    print("=" * 75)

    if not image_files:
        print("[Notice] Drop 15-20 classroom photos into datasets/campus_test_set/ to run full evaluation.")
        return

    total_time = 0.0
    detected_count = 0
    rectified_count = 0

    print(f"{'Image Name':<25} | {'Detected':<10} | {'Conf (%)':<10} | {'Homography':<12} | {'Latency (ms)':<12}")
    print("-" * 75)

    for img_path in image_files:
        img = Image.open(img_path)
        t0 = time.perf_counter()

        detections, primary_box, annotated_pil = detector.detect(img, conf_threshold=0.25)
        rectified_pil, is_rectified, corners = preprocessor.rectify_whiteboard(img, primary_box)

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        total_time += t_elapsed

        detected = primary_box is not None
        conf = (primary_box.confidence * 100.0) if detected else 0.0

        if detected:
            detected_count += 1
        if is_rectified:
            rectified_count += 1

        # Save annotated detection & rectified crop
        annotated_pil.save(out_dir / f"det_{img_path.name}")
        rectified_pil.save(out_dir / f"rect_{img_path.name}")

        rect_str = "Applied" if is_rectified else "Axis-Crop"
        print(f"{img_path.name:<25} | {str(detected):<10} | {conf:<10.1f} | {rect_str:<12} | {t_elapsed:<12.1f}")

    avg_latency = total_time / len(image_files)
    det_rate = (detected_count / len(image_files)) * 100.0
    rect_rate = (rectified_count / len(image_files)) * 100.0

    print("=" * 75)
    print("DOMAIN TRANSFER SUMMARY:")
    print(f"  • Detection Rate:            {det_rate:.1f}% ({detected_count}/{len(image_files)})")
    print(f"  • 4-Point Homography Rate:   {rect_rate:.1f}% ({rectified_count}/{len(image_files)})")
    print(f"  • Mean Inference Latency:    {avg_latency:.1f} ms/image")
    print(f"  • Visualized outputs saved:  {out_dir}")
    print("=" * 75)


if __name__ == "__main__":
    run_campus_evaluation()
