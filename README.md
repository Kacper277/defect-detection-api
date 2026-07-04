# Defect Detection API

A REST API for automated steel surface defect detection using **ResNet-18** fine-tuned on the [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) dataset. Includes Grad-CAM heatmap visualization and real-time data drift monitoring.

---

## Features

- **6-class defect classification** — crazing, inclusion, patches, pitted surface, rolled-in scale, scratches
- **Grad-CAM heatmaps** — visualize which regions of the image triggered the prediction
- **Data drift monitoring** — tracks pixel statistics and prediction confidence over time
- **Streamlit dashboard** — interactive monitoring UI
- **Docker support** — containerized for local or cloud deployment
- **Azure deployment** — App Service, ACI, or AKS

---

## Project Structure

```
defect-detection-api/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   └── model_loader.py      # ModelService, GradCAM, singleton
├── train/
│   ├── config.py            # Paths, hyperparameters, class definitions
│   ├── dataset.py           # NEUDETDataset + Albumentations transforms
│   └── train.py             # Training loop, checkpointing, metrics
├── monitoring/
│   ├── data_drift.py        # Drift detection, prediction logging
│   ├── dashboard.py         # Streamlit monitoring dashboard
│   └── requirements.txt     # Monitoring-specific dependencies
├── deploy/
│   ├── azure-deployment.md  # Detailed Azure deployment guide
│   ├── deploy_to_azure.ps1  # One-click PowerShell deployment script
│   └── kubernetes.yaml      # AKS Deployment + Service + HPA
├── checkpoints/             # Saved model weights (best_model.pth)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── startup.sh               # Azure App Service startup script
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Download the NEU-DET dataset (auto-downloaded via `kagglehub`) and run:

```bash
python -m train.train
```

This saves `checkpoints/best_model.pth` and computes the drift baseline.

### 3. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with gunicorn:

```bash
gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers 1
```

### 4. Open Swagger UI

```
http://localhost:8000/docs
```

---

## Docker

### Build & run

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### With monitoring dashboard

```bash
docker compose --profile monitoring up
```

Dashboard available at `http://localhost:8501`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model status and loaded classes |
| `GET` | `/classes` | List of detectable defect classes |
| `POST` | `/predict` | Predict defect class from image |
| `POST` | `/predict/with-heatmap-image` | Predict and return Grad-CAM overlay as JPEG |

### Example: predict

```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@sample.jpg" \
    -F "include_heatmap=false"
```

Response:

```json
{
  "predicted_class": "crazing",
  "predicted_index": 0,
  "confidence": 0.9741,
  "all_probabilities": {
    "crazing": 0.9741,
    "inclusion": 0.0102,
    "patches": 0.0057,
    "pitted_surface": 0.0048,
    "rolled-in_scale": 0.0031,
    "scratches": 0.0021
  },
  "heatmap_shape": [224, 224],
  "heatmap": null
}
```

### Example: predict with heatmap overlay image

```bash
curl -X POST http://localhost:8000/predict/with-heatmap-image \
    -F "file=@sample.jpg" \
    --output heatmap_overlay.jpg
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Fine-tuned layers | `layer4` + classifier head |
| Input size | 224 × 224 RGB |
| Output | 6-class softmax |
| Dropout | 0.3 before final FC |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Early stopping | patience=7 epochs |

---

## Training Configuration

Edit `train/config.py` to adjust:

```python
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = "cpu"   # change to "cuda" for GPU
SEED = 42
```

---

## Monitoring

After training, a baseline distribution is automatically computed from the training set. The API logs every prediction to `monitoring/logs/predictions_YYYYMM.jsonl` and flags drift when:

- Mean pixel deviates **> 3 sigma** from the training baseline
- Prediction confidence drops **below 0.5**

Alerts are saved to `monitoring/logs/drift_alerts.jsonl`.

### Run the dashboard

```bash
streamlit run monitoring/dashboard.py
```

---

## Using the Hosted API

The API is publicly deployed on Azure and can be called directly — no local setup, Docker, or PyTorch installation required. All you need is the `requests` library.

```bash
pip install requests
```

### Health check

```python
import requests

response = requests.get("https://defect-detection-api.azurewebsites.net/health")
print(response.json())
```

### Predict defect class

```python
import requests

with open("sample.jpg", "rb") as f:
    response = requests.post(
        "https://defect-detection-api.azurewebsites.net/predict",
        files={"file": f},
        data={"include_heatmap": "false"},
    )

result = response.json()
print(f"Class:      {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probs:  {result['all_probabilities']}")
```

### Predict and save Grad-CAM overlay image

```python
import requests

with open("sample.jpg", "rb") as f:
    response = requests.post(
        "https://defect-detection-api.azurewebsites.net/predict/with-heatmap-image",
        files={"file": f},
    )

with open("heatmap_overlay.jpg", "wb") as out:
    out.write(response.content)

print("Heatmap saved to heatmap_overlay.jpg")
```

> **Note:** The free tier (F1) has a cold start of ~30–40 seconds after inactivity. If the first request times out, retry after a moment.

---

## Azure Deployment

See [`deploy/azure-deployment.md`](deploy/azure-deployment.md) for the full guide. Summary of options:

| Method | Difficulty | Est. Time | Cost |
|--------|:----------:|:---------:|------|
| App Service (source code) | Low | 10 min | Free tier (F1) |
| Azure Container Instances | Low | 10 min | ~$30/month |
| App Service (container) | Medium | 15 min | ~$45/month (B1) |
| AKS | High | 30 min | Variable |

### One-click deploy (PowerShell)

```powershell
.\deploy\deploy_to_azure.ps1
```

### Smoke test after deployment

```bash
curl https://defect-detection-api.azurewebsites.net/health
```

---

## Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `crazing` | Network of fine surface cracks |
| 1 | `inclusion` | Embedded foreign material |
| 2 | `patches` | Irregular surface patches |
| 3 | `pitted_surface` | Small pits or holes |
| 4 | `rolled-in_scale` | Scale pressed into surface during rolling |
| 5 | `scratches` | Linear surface scratches |

---

## Requirements

- Python 3.11
- PyTorch 2.0.1 (CPU build)
- FastAPI + Uvicorn + Gunicorn
- OpenCV (headless)
- Albumentations
- scikit-learn
- Streamlit + Plotly (monitoring only)

---

## Cleanup (Azure)

To remove all Azure resources and stop billing:

```bash
az group delete --name defect-detection-rg --yes --no-wait
```
#
