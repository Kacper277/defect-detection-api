"""
FastAPI REST API for production defect detection.

Endpoints:
- POST /predict: accepts an image, returns prediction + optional Grad-CAM heatmap
- GET /health: health check
- GET /classes: list of classes
"""
import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

# Add root directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model_loader import get_model_service
from monitoring.data_drift import log_prediction, detect_drift

app = FastAPI(
    title="Defect Detection API",
    description="API for production defect detection on images (NEU-DET). "
                "Uses ResNet-18 + Grad-CAM.",
    version="1.0.0",
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    all_probabilities: dict[str, float]
    heatmap_shape: list[int]
    heatmap: Optional[list[list[float]]] = None  # now optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


@app.on_event("startup")
def startup():
    """Initialize model on startup."""
    print("Loading model...")
    try:
        get_model_service()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")


@app.get("/", include_in_schema=False)
async def root():
    """Redirects to Swagger UI documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and model status."""
    try:
        service = get_model_service()
        model_loaded = True
        classes = service.classes
    except Exception:
        model_loaded = False
        classes = []

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        classes=classes,
    )


@app.get("/classes")
def get_classes():
    """Returns list of detectable defect classes."""
    try:
        service = get_model_service()
        return {"classes": service.classes, "count": len(service.classes)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    include_heatmap: bool = False,  # default False – faster
):
    """
    Performs defect prediction on uploaded image.

    Args:
        file: image in JPEG, PNG, BMP, TIFF format
        include_heatmap: whether to return Grad-CAM heatmap (large response!)

    Returns:
        predicted_class: name of detected defect
        confidence: prediction confidence (0-1)
        all_probabilities: probabilities for all classes
        heatmap: (optional) 224x224 matrix of activation values
    """
    # File validation
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Expected an image.",
        )

    # Load image
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)} bytes). Maximum size: {MAX_IMAGE_SIZE} bytes.",
        )

    if len(contents) < 100:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty or too small.",
        )

    # Perform prediction
    try:
        service = get_model_service()
        result = service.predict_bytes(contents)

        # Monitoring: log prediction and detect drift
        try:
            log_prediction(result, contents)
            drift = detect_drift({
                "mean_pixel": result.get("_mean_pixel", 128),  # now works
                "confidence": result["confidence"]
            })
            if drift.get("drift_detected"):
                print(f"[DRIFT ALERT] {drift}")
        except Exception as e:
            print(f"[MONITORING WARN] {e}")

        # Prepare response
        response = {
            "predicted_class": result["predicted_class"],
            "predicted_index": result["predicted_index"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "heatmap_shape": list(result["heatmap_shape"]),
        }

        # Conditional heatmap inclusion
        if include_heatmap:
            response["heatmap"] = result["heatmap"]

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained. {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


@app.post("/predict/with-heatmap-image")
async def predict_with_heatmap_image(file: UploadFile = File(...)):
    """
    Performs prediction and returns image with Grad-CAM heatmap overlay.
    """
    contents = await file.read()

    try:
        service = get_model_service()
        result = service.predict_bytes(contents)

        # Get original image
        image_array = np.frombuffer(contents, np.uint8)
        original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Prepare visualization: heatmap on original image
        heatmap = np.array(result["heatmap"], dtype=np.float32)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Resize heatmap to original image size
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(heatmap_colored, (w, h))

        # Overlay heatmap on original
        overlay = cv2.addWeighted(original, 0.6, heatmap_resized, 0.4, 0)

        # Add label
        label = f"{result['predicted_class']} ({result['confidence']:.1%})"
        cv2.putText(
            overlay, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # Return as JPEG
        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)