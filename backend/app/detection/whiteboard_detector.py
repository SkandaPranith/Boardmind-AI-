"""
backend/app/detection/whiteboard_detector.py
Whiteboard Detection Service using Ultralytics YOLO on CPU.
Loads champion model once at initialization.
"""

import io
import base64
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from backend.app.schemas.models import BoundingBox

class WhiteboardDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cpu'
        self.model_path = self._resolve_model_path(model_path)
        print(f"[Detector] Initializing WhiteboardDetector with weights: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: 'Whiteboard'}
        print(f"[Detector] WhiteboardDetector successfully loaded on {self.device}. Classes: {self.class_names}")

    def _resolve_model_path(self, user_path: Optional[str]) -> Path:
        if user_path and Path(user_path).exists():
            return Path(user_path).resolve()

        # Robust root resolution: walk up parent hierarchy until repo root is found
        curr = Path(__file__).resolve().parent
        project_root = None
        for p in [curr] + list(curr.parents):
            if (p / "models" / "whiteboard_detector_best.pt").exists() or (p / "datasets").exists():
                project_root = p
                break
        
        if project_root is None:
            # Fallback: 4 levels up from backend/app/detection/whiteboard_detector.py -> repo root
            project_root = Path(__file__).resolve().parent.parent.parent.parent

        candidates = [
            project_root / "models" / "whiteboard_detector_best.pt",
            project_root / "runs" / "whiteboard_optimization" / "exp0_baseline_yolov8n" / "weights" / "best.pt",
            project_root / "runs" / "whiteboard_optimization" / "exp1_aug_perspective_scale" / "weights" / "best.pt",
            project_root / "runs" / "whiteboard_detection_test" / "weights" / "best.pt",
            Path("models/whiteboard_detector_best.pt"),
            Path("../models/whiteboard_detector_best.pt"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError(
            f"No trained whiteboard detection checkpoint found. Checked candidates: {[str(c) for c in candidates]}"
        )

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6,
        conf_board: Optional[float] = None,
        conf_text: Optional[float] = None
    ) -> Tuple[List[BoundingBox], Optional[BoundingBox], Image.Image]:
        """
        Runs whiteboard and handwritten text detection inference on a PIL image.
        Supports dual class-specific confidence thresholds:
            - conf_board: Confidence threshold specifically for Whiteboard (e.g. 0.50)
            - conf_text: Confidence threshold specifically for Handwritten text (e.g. 0.15)
            - conf_threshold: Fallback threshold if specific class thresholds are not set
        """
        c_board = conf_board if conf_board is not None else conf_threshold
        c_text = conf_text if conf_text is not None else conf_threshold
        min_conf = min(c_board, c_text, conf_threshold)

        orig_w, orig_h = image.size
        # Run inference using the lowest requested threshold so no valid candidates are dropped early
        results = self.model.predict(
            source=image,
            conf=min_conf,
            iou=iou_threshold,
            device=self.device,
            imgsz=416,
            verbose=False
        )[0]

        detections: List[BoundingBox] = []
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                conf = float(confs[i])
                cls_id = int(clss[i])
                cls_name = self.class_names.get(cls_id, "Whiteboard")
                is_whiteboard = ("whiteboard" in cls_name.lower()) or (cls_id == 1)

                # Class-specific threshold gating
                required_conf = c_board if is_whiteboard else c_text
                if conf < required_conf:
                    continue

                xmin, ymin, xmax, ymax = boxes[i].tolist()

                # Clamp to image boundaries
                xmin = max(0.0, min(xmin, float(orig_w)))
                ymin = max(0.0, min(ymin, float(orig_h)))
                xmax = max(0.0, min(xmax, float(orig_w)))
                ymax = max(0.0, min(ymax, float(orig_h)))

                norm_box = [
                    round(xmin / orig_w, 4),
                    round(ymin / orig_h, 4),
                    round(xmax / orig_w, 4),
                    round(ymax / orig_h, 4)
                ]

                bbox = BoundingBox(
                    xmin=round(xmin, 1),
                    ymin=round(ymin, 1),
                    xmax=round(xmax, 1),
                    ymax=round(ymax, 1),
                    normalized=norm_box,
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(bbox)

                # Color-code by class: Cyan (#0284c7) for Whiteboard, Orange (#ea580c) for Text
                box_color = "#0284c7" if is_whiteboard else "#ea580c"

                # Draw high-contrast bounding box on image
                draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=4)
                # Label banner
                label = f"{cls_name} {conf:.1%}"
                text_bbox = draw.textbbox((xmin + 4, max(0.0, ymin - 20)), label)
                draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill=box_color)
                draw.text((xmin + 4, max(0.0, ymin - 20)), label, fill="white")

        # Select primary detection (prioritize whiteboard class, confidence and size)
        primary_box = None
        if detections:
            # Filter for Whiteboard class (name 'Whiteboard' or class_id 1)
            wb_detections = [b for b in detections if 'whiteboard' in b.class_name.lower() or b.class_id == 1]
            pool = wb_detections if wb_detections else detections

            # Sort by area * confidence
            def score(b: BoundingBox):
                area = (b.xmax - b.xmin) * (b.ymax - b.ymin)
                return area * (b.confidence ** 2)
            primary_box = max(pool, key=score)

        return detections, primary_box, annotated_image

    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        buffered = io.BytesIO()
        if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format=format, quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

# Global singleton detector instance
_detector_instance: Optional[WhiteboardDetector] = None

def get_whiteboard_detector(model_path: Optional[str] = None, force_reload: bool = False) -> WhiteboardDetector:
    global _detector_instance
    if _detector_instance is None or force_reload or (model_path is not None):
        _detector_instance = WhiteboardDetector(model_path=model_path)
    return _detector_instance

