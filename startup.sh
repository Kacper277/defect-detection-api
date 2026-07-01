#!/bin/bash
set -e

echo "Starting Defect Detection API deployment..."
echo "Python version: $(python --version)"

# Install dependencies (Azure Oryx already ran pip, but double-check)
pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Verify model checkpoint exists
if [ -f "checkpoints/best_model.pth" ]; then
    echo "Model checkpoint found: $(ls -lh checkpoints/best_model.pth | awk '{print $5}')"
else
    echo "WARNING: Model checkpoint not found. API will start but predictions will fail."
fi

# Start the API
echo "Starting API server..."
exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile -