"""
FastAPI REST API dla detekcji wad produkcyjnych.

Endpoints:
- POST /predict: przyjmuje obraz, zwraca predykcję + heatmapę Grad-CAM
- GET /health: health check
- GET /classes: lista klas
"""
import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# Dodaj katalog główny do ścieżki
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model_loader import get_model_service
from monitoring.data_drift import log_prediction, detect_drift

app = FastAPI(
    title="Defect Detection API",
    description="API do detekcji wad produkcyjnych na obrazach (NEU-DET). "
                "Wykorzystuje ResNet-18 + Grad-CAM.",
    version="1.0.0",
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    all_probabilities: dict[str, float]
    heatmap_shape: list[int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]


@app.on_event("startup")
def startup():
    """Inicjalizacja modelu przy starcie."""
    print("Ładowanie modelu...")
    try:
        get_model_service()
        print("Model załadowany pomyślnie!")
    except Exception as e:
        print(f"BŁĄD ładowania modelu: {e}")
        # Nie przerywamy startu - API będzie działać,
        # ale /predict zwróci 503


# ============================================================
# NOWY ENDPOINT: Strona główna przekierowująca do /docs
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Strona główna API – wyświetla podstawowe informacje i link do dokumentacji.
    Możesz też od razu przekierować do /docs (zakomentowana alternatywa).
    """
    # Alternatywa: natychmiastowe przekierowanie
    # return RedirectResponse(url="/docs")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API do detekcji wad produkcyjnych</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 60px; background: #f5f5f5; }
            .container { max-width: 700px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .btn { display: inline-block; padding: 12px 24px; background: #0066cc; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }
            .btn:hover { background: #0052a3; }
            code { background: #eee; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 API do detekcji wad produkcyjnych</h1>
            <p>Wykrywanie defektów na zdjęciach z użyciem modelu <strong>ResNet-18</strong> i wizualizacji <strong>Grad-CAM</strong>.</p>
            <p>Dostępne endpointy:</p>
            <ul>
                <li><code>/health</code> – status serwera i modelu</li>
                <li><code>/classes</code> – lista klas defektów</li>
                <li><code>/predict</code> – predykcja dla przesłanego obrazu</li>
            </ul>
            <p>
                <a href="/docs" class="btn">📖 Dokumentacja Swagger UI</a>
                <a href="/redoc" class="btn">📄 Dokumentacja ReDoc</a>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Sprawdzenie stanu API i modelu."""
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
    """Zwraca listę klas wykrywanych defektów."""
    try:
        service = get_model_service()
        return {"classes": service.classes, "count": len(service.classes)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model nie załadowany: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    include_heatmap: bool = True,
):
    """
    Wykonuje predykcję defektu na przesłanym obrazie.

    Args:
        file: obraz w formacie JPEG, PNG, BMP, TIFF
        include_heatmap: czy zwrócić heatmapę Grad-CAM

    Returns:
        predicted_class: nazwa wykrytego defektu
        confidence: pewność predykcji (0-1)
        all_probabilities: prawdopodobieństwa dla wszystkich klas
        heatmap: (opcjonalnie) macierz 224x224 z wartościami aktywacji
    """
    # Walidacja pliku
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Nieobsługiwany typ pliku: {file.content_type}. Oczekiwano obrazu.",
        )

    # Wczytaj obraz
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Plik zbyt duży ({len(contents)} bytes). Maksymalny rozmiar: {MAX_IMAGE_SIZE} bytes.",
        )

    if len(contents) < 100:
        raise HTTPException(
            status_code=400,
            detail="Przesłany plik jest pusty lub zbyt mały.",
        )

        # Wykonaj predykcję
    try:
        service = get_model_service()
        result = service.predict_bytes(contents)

        # Monitoring: loguj predykcję i wykrywaj dryf
        try:
            log_prediction(result, contents)
            drift = detect_drift({"mean_pixel": result.get("_mean_pixel", 128),
                                  "confidence": result["confidence"]})
            if drift.get("drift_detected"):
                print(f"[DRIFT ALERT] {drift}")
        except Exception as e:
            print(f"[MONITORING WARN] {e}")

        # Przygotuj odpowiedź
        response = {
            "predicted_class": result["predicted_class"],
            "predicted_index": result["predicted_index"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "heatmap_shape": list(result["heatmap_shape"]),
        }

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model nie został wytrenowany. {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Błąd podczas predykcji: {str(e)}",
        )


@app.post("/predict/with-heatmap-image")
async def predict_with_heatmap_image(file: UploadFile = File(...)):
    """
    Wykonuje predykcję i zwraca obraz z nałożoną heatmapą Grad-CAM.
    """
    contents = await file.read()

    try:
        service = get_model_service()
        result = service.predict_bytes(contents)

        # Pobierz oryginalny obraz
        image_array = np.frombuffer(contents, np.uint8)
        original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Przygotuj wizualizację: heatmapa na oryginalnym obrazie
        heatmap = np.array(result["heatmap"], dtype=np.float32)
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Zmień rozmiar heatmapy do rozmiaru oryginalnego obrazu
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(heatmap_colored, (w, h))

        # Nałóż heatmapę na oryginał
        overlay = cv2.addWeighted(original, 0.6, heatmap_resized, 0.4, 0)

        # Dodaj etykietę
        label = f"{result['predicted_class']} ({result['confidence']:.1%})"
        cv2.putText(
            overlay, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # Zwróć jako JPEG
        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
