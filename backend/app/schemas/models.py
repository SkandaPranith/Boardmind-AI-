"""
backend/app/schemas/models.py
Pydantic Schemas for BoardMind AI REST API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    xmin: float = Field(..., description="Top-left X coordinate in pixels")
    ymin: float = Field(..., description="Top-left Y coordinate in pixels")
    xmax: float = Field(..., description="Bottom-right X coordinate in pixels")
    ymax: float = Field(..., description="Bottom-right Y coordinate in pixels")
    normalized: Optional[List[float]] = Field(None, description="[xmin, ymin, xmax, ymax] normalized to 0.0-1.0")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    class_id: int = Field(0, description="Class ID (0 for Whiteboard)")
    class_name: str = Field("Whiteboard", description="Detected class name")

class DetectResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[BoundingBox]
    primary_detection: Optional[BoundingBox] = None
    image_width: int
    image_height: int
    annotated_image: Optional[str] = Field(None, description="Base64 data URL of image with bounding boxes drawn")
    message: str

class RecognizedLine(BaseModel):
    line_number: int
    text: str
    confidence: float
    bbox: Optional[List[float]] = None

class ExtractTextResponse(BaseModel):
    success: bool
    model_name: str
    status: str = Field(..., description="Model state: ready, untrained_baseline, or offline")
    raw_text: str
    lines: List[RecognizedLine] = []
    stroke_density: Optional[float] = Field(None, description="Measured percentage of ink stroke pixels in crop")
    message: str

class GenerateNotesRequest(BaseModel):
    raw_text: str
    topic_hint: Optional[str] = None
    api_key: Optional[str] = None

class StructuredNotes(BaseModel):
    title: str
    headings: List[str] = []
    bullet_points: List[str] = []
    numbered_steps: List[str] = []
    definitions: List[str] = []
    formulas_or_key_terms: List[str] = []
    raw_preserved_text: str
    markdown_content: str
    ai_concept_explanation: Optional[str] = None
    ai_engine_used: Optional[str] = None

class CVPipelineMetadata(BaseModel):
    perspective_rectified: bool = Field(False, description="Whether 4-corner homography transform was applied")
    corners_detected: Optional[List[List[float]]] = Field(None, description="Detected 4-corner quad coordinates [[x,y],...]")
    stroke_density_percent: float = Field(0.0, description="Measured percentage of ink stroke pixels via Otsu threshold")
    segmented_lines_count: int = Field(0, description="Number of text lines segmented via horizontal projection profiling")
    yolo_model_params: str = Field("3.14M (YOLOv8n)", description="Detection backbone parameter count")
    detection_losses: str = Field("CIoU (Overlap/Aspect) + DFL (Distribution Focal) + BCE (Classification)", description="Mathematical loss functions used")
    evaluation_test_mAP50: str = Field("99.50%", description="Test set mAP@50 score")
    evaluation_test_mAP50_95: str = Field("91.70%", description="Test set mAP@50-95 score")

class GenerateNotesResponse(BaseModel):
    success: bool
    notes: StructuredNotes
    message: str

class ProcessResponse(BaseModel):
    success: bool
    detection: DetectResponse
    cropped_whiteboard: Optional[str] = Field(None, description="Base64 data URL of cropped, preprocessed whiteboard")
    cv_diagnostic_image: Optional[str] = Field(None, description="Base64 diagnostic image showing line segmentation cuts")
    binary_mask_image: Optional[str] = Field(None, description="Base64 binary mask of Otsu ink strokes")
    cv_metadata: Optional[CVPipelineMetadata] = None
    text_extraction: ExtractTextResponse
    structured_notes: StructuredNotes
    ai_concept_explanation: Optional[str] = None
    processing_time_ms: float
    warnings: List[str] = []

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    whiteboard_detector: Dict[str, Any]
    handwriting_recognizer: Dict[str, Any]
    ollama: Dict[str, Any]
    hardware: Dict[str, Any]

