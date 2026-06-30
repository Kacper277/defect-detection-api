# ============================================================
# Dockerfile — Defect Detection API
# Wieloetapowy build: Python 3.11 + PyTorch + FastAPI
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalujemy tylko niezbędne narzędzia systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy requirements
COPY requirements.txt .

# Instalujemy zależności
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Biblioteki systemowe dla OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy zainstalowane pakiety z buildera
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Kopiujemy kod aplikacji (bez danych treningowych)
COPY app/ ./app/
COPY train/config.py ./train/config.py
COPY checkpoints/ ./checkpoints/

# Zmienna środowiskowa
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Port API
EXPOSE 8000

# Uruchom API przez gunicorn z uvicorn workerami
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
