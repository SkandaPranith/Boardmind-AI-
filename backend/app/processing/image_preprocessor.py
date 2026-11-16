"""
backend/app/processing/image_preprocessor.py
Image Preprocessing, Perspective Rectification, and Whiteboard Region Enhancement.
Provides:
- 4-Corner Homography Perspective Correction (cv2.getPerspectiveTransform & warpPerspective)
- Accurate Bounding-box cropping with safe margin padding
- Contrast enhancement (CLAHE in LAB space)
- Noise reduction preserving handwritten stroke integrity
- Ink stroke density measurement (Otsu thresholding)
- Horizontal projection profiling for line segmentation
"""

import io
import base64
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2
from PIL import Image

from backend.app.schemas.models import BoundingBox

class ImagePreprocessor:
    def __init__(self, padding_ratio: float = 0.02):
        self.padding_ratio = padding_ratio

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates: [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def rectify_whiteboard(
        self,
        image: Image.Image,
        bbox: Optional[BoundingBox] = None
    ) -> Tuple[Image.Image, bool, Optional[List[List[float]]]]:
        """
        Detects whiteboard 4-corner polygon within the detected bounding box
        and applies projective homography (cv2.getPerspectiveTransform + warpPerspective).
        Falls back to padded bounding-box crop if no clean quadrilateral is found.
        """
        w, h = image.size
        cv_orig = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 1. Obtain region of interest (ROI)
        if bbox is not None:
            pad_x = (bbox.xmax - bbox.xmin) * self.padding_ratio
            pad_y = (bbox.ymax - bbox.ymin) * self.padding_ratio
            x1 = max(0, int(bbox.xmin - pad_x))
            y1 = max(0, int(bbox.ymin - pad_y))
            x2 = min(w, int(bbox.xmax + pad_x))
            y2 = min(h, int(bbox.ymax + pad_y))
            roi = cv_orig[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            roi = cv_orig
            offset_x, offset_y = 0, 0

        roi_h, roi_w = roi.shape[:2]
        if roi_h < 30 or roi_w < 30:
            return image.copy(), False, None

        # 2. Contour & Polygon Approximation for Perspective Quad
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        quad_found = False
        ordered_pts = None

        min_area = (roi_w * roi_h) * 0.35  # At least 35% of detected board area

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype("float32")
                ordered_pts = self.order_points(pts)
                quad_found = True
                break

        # 3. Apply Homography Warp if Quad Found
        if quad_found and ordered_pts is not None:
            (tl, tr, br, bl) = ordered_pts

            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_w = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_h = max(int(height_a), int(height_b))

            if max_w > 50 and max_h > 50:
                dst = np.array([
                    [0, 0],
                    [max_w - 1, 0],
                    [max_w - 1, max_h - 1],
                    [0, max_h - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(ordered_pts, dst)
                warped = cv2.warpPerspective(roi, M, (max_w, max_h))
                warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

                # Global coordinates for visualization
                global_corners = [
                    [round(float(pt[0] + offset_x), 1), round(float(pt[1] + offset_y), 1)]
                    for pt in ordered_pts
                ]
                return warped_pil, True, global_corners

        # 4. Fallback: Clean axis-aligned crop
        fallback_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return fallback_pil, False, None

    def crop_whiteboard(
        self,
        image: Image.Image,
        bbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Standard crop fallback."""
        crop, _, _ = self.rectify_whiteboard(image, bbox)
        return crop

    def enhance_handwriting(
        self,
        image: Image.Image,
        enhance_contrast: bool = True,
        denoise: bool = True,
        reconnect_strokes: bool = True
    ) -> Tuple[Image.Image, float, Image.Image]:
        """
        Enhanced multi-stage handwriting preprocessor:
        1. Bilateral filter for edge-preserving denoising
        2. High-contrast CLAHE in LAB space (clipLimit=3.0) for faint dry-erase markers
        3. Morphological closing (2x2) to reconnect broken pen skips
        4. Otsu stroke density measurement
        5. Returns: (enhanced_pil, stroke_density, binary_mask_pil)
        """
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 1. Bilateral filter for edge-preserving denoising
        if denoise:
            cv_img = cv2.bilateralFilter(cv_img, d=5, sigmaColor=50, sigmaSpace=50)

        # 2. CLAHE on Luminance channel
        if enhance_contrast:
            lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            cv_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 3. Morphological stroke bridge to reconnect skipped marker lines
        if reconnect_strokes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_CLOSE, kernel)

        # 4. Calculate stroke density via Otsu inverse threshold on grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        stroke_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        stroke_density = (stroke_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        # Convert back to PIL RGB
        enhanced_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        binary_pil = Image.fromarray(thresh)
        return enhanced_pil, round(stroke_density, 2), binary_pil

    def compute_horizontal_projection_lines(
        self,
        image: Image.Image
    ) -> Tuple[List[Tuple[int, int]], Image.Image]:
        """
        Performs line segmentation using Horizontal Ink Projection Profiling.
        Returns line boundary slices (y_start, y_end) and an annotated diagnostic image.
        """
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_proj = np.sum(binary == 255, axis=1)

        # 1D smoothing
        kernel_size = max(5, int(h * 0.02))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(horizontal_proj.astype(np.float32).reshape(-1, 1), (1, kernel_size), 0).flatten()

        thresh_val = np.mean(smoothed) * 0.15
        line_indices: List[Tuple[int, int]] = []
        in_line = False
        start_y = 0

        for y, val in enumerate(smoothed):
            if val > thresh_val and not in_line:
                in_line = True
                start_y = y
            elif val <= thresh_val and in_line:
                in_line = False
                if (y - start_y) > max(10, int(h * 0.03)):
                    line_indices.append((start_y, y))

        if in_line and (h - start_y) > max(10, int(h * 0.03)):
            line_indices.append((start_y, h))

        # Create diagnostic visualization showing line segmentation cuts
        diag_img = cv_img.copy()
        for idx, (y1, y2) in enumerate(line_indices):
            cv2.rectangle(diag_img, (0, y1), (w - 1, y2), (0, 210, 255), 2)
            cv2.putText(diag_img, f"L{idx+1}", (10, min(h - 5, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return line_indices, Image.fromarray(cv2.cvtColor(diag_img, cv2.COLOR_BGR2RGB))

    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        buffered = io.BytesIO()
        if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format=format, quality=92)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

_preprocessor_instance: Optional[ImagePreprocessor] = None

def get_image_preprocessor() -> ImagePreprocessor:
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = ImagePreprocessor()
    return _preprocessor_instance


