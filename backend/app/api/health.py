"""
backend/app/api/health.py
Health and System Status API Endpoint.
"""

import platform
import psutil
import urllib.request
import urllib.error
import json
import os
from fastapi import APIRouter
from backend.app.schemas.models import HealthResponse
from backend.app.detection.whiteboard_detector import get_whiteboard_detector
from backend.app.recognition.handwriting_recognizer import get_handwriting_recognizer
from backend.app.processing.ai_explainer import get_ai_concept_explainer

router = APIRouter(prefix="/api", tags=["Health"])


async def check_ollama_status() -> dict:
    """Check Ollama service status and available models."""
    explainer = get_ai_concept_explainer()
    host = explainer.ollama_host
    model = explainer.ollama_model

    # Check service availability
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                model_available = any(model in m or m.startswith(model.split(":")[0]) for m in models)
                return {
                    "available": True,
                    "host": host,
                    "configured_model": model,
                    "model_available": model_available,
                    "models": models
                }
    except Exception as e:
        pass

    return {
        "available": False,
        "host": host,
        "configured_model": model,
        "model_available": False,
        "models": [],
        "error": "Ollama service not reachable"
    }


@router.get("/health", response_model=HealthResponse)
async def get_health():
    detector = get_whiteboard_detector()
    recognizer = get_handwriting_recognizer()
    ollama_status = await check_ollama_status()

    return HealthResponse(
        status="healthy",
        app_name="BoardMind AI",
        version="1.0.0",
        whiteboard_detector={
            "loaded": True,
            "device": detector.device,
            "weights_path": str(detector.model_path),
            "classes": detector.class_names
        },
        handwriting_recognizer=recognizer.get_metadata(),
        ollama=ollama_status,
        hardware={
            "os": platform.system(),
            "cpu_architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
            "ram_used_percent": psutil.virtual_memory().percent
        }
    )


@router.get("/health/ollama")
async def get_ollama_health():
    """Dedicated Ollama health check endpoint."""
    return await check_ollama_status()

