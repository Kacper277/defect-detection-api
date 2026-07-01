# Multi-stage build for Defect Detection API
# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# System libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code (without training data)
COPY app/ ./app/
COPY train/config.py ./train/config.py
COPY checkpoints/ ./checkpoints/
COPY monitoring/ ./monitoring/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# API port
EXPOSE 8000

# Run API via gunicorn with uvicorn workers
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]