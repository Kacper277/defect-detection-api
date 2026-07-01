````markdown
# Defect Detection API – Azure Deployment Guide

## Deployment options overview

| Method | Requires Docker | Difficulty | Estimated Time |
|--------|:---------------:|:----------:|---------------:|
| App Service from source code | No | Low | 10 min |
| App Service from container | Yes | Medium | 15 min |
| Azure Container Instances (ACI) | Yes | Low | 10 min |
| Azure Kubernetes Service (AKS) | Yes | High | 30 min |

---

# Option A: App Service from source code (No Docker) **[RECOMMENDED]**

This deployment method does **not** require Docker. Azure automatically builds the application environment using the project's `requirements.txt` and `startup.sh`.

## Prerequisites

Install Azure CLI (one-time):

```powershell
winget install -e --id Microsoft.AzureCLI
```

Login to Azure:

```powershell
az login
```

Verify that your Azure subscription is active:

```powershell
az account show
```

A free Azure account with the **$200 credit** is sufficient for this deployment.

---

## Step-by-step deployment

### 1. Create a Resource Group

```bash
az group create \
    --name defect-detection-rg \
    --location westeurope
```

### 2. Create an App Service Plan (Linux)

```bash
az appservice plan create \
    --name defect-detection-plan \
    --resource-group defect-detection-rg \
    --sku F1 \
    --is-linux
```

### 3. Create the Web App

```bash
az webapp create \
    --name defect-detection-api \
    --resource-group defect-detection-rg \
    --plan defect-detection-plan \
    --runtime "PYTHON:3.11"
```

### 4. Configure the startup command

```bash
az webapp config set \
    --name defect-detection-api \
    --resource-group defect-detection-rg \
    --startup-file "startup.sh" \
    --always-on false
```

### 5. Configure environment variables

```bash
az webapp config appsettings set \
    --name defect-detection-api \
    --resource-group defect-detection-rg \
    --settings \
        PYTHONPATH="/home/site/wwwroot" \
        DEVICE="cpu" \
        PYTHONUNBUFFERED="1"
```

### 6. Deploy the application

Navigate to the project directory:

```bash
cd C:\Users\Kacper\Desktop\defect-detection-api
```

Deploy the application:

```bash
az webapp up \
    --name defect-detection-api \
    --resource-group defect-detection-rg \
    --runtime "PYTHON:3.11" \
    --os-type linux
```

### 7. Verify deployment

```bash
curl https://defect-detection-api.azurewebsites.net/health
```

---

## One-click deployment (PowerShell)

If your repository contains the deployment script:

```powershell
.\deploy\deploy_to_azure.ps1
```

The script automatically:

- verifies Azure CLI installation,
- authenticates with Azure,
- creates required resources,
- deploys the application,
- validates the deployment.

---

## Free Tier (F1) limitations

| Feature | Limitation |
|---------|------------|
| Compute | 60 minutes/day |
| Memory | 1 GB RAM |
| Storage | 1 GB |
| Cold Start | ~30–40 seconds after inactivity |
| Custom Domains | Not supported |
| Auto Scaling | Not available |

The trained model (~200 MB) comfortably fits within the free memory allocation.

---

## Smoke Tests

### Health endpoint

```bash
curl https://defect-detection-api.azurewebsites.net/health
```

### Prediction (without heatmap)

```bash
curl -X POST https://defect-detection-api.azurewebsites.net/predict \
    -F "file=@test_image.jpg" \
    -F "include_heatmap=false"
```

### Prediction with heatmap image

```bash
curl -X POST https://defect-detection-api.azurewebsites.net/predict/with-heatmap-image \
    -F "file=@test_image.jpg" \
    --output heatmap.jpg
```

### Swagger UI

Open:

```text
https://defect-detection-api.azurewebsites.net/docs
```

---

# Option B: Azure Container Instances (ACI)

This deployment method requires:

- Docker Desktop
- Azure CLI
- Active Azure Subscription

---

## 1. Login

```bash
az login
```

---

## 2. Create Azure Container Registry

```bash
az acr create \
    --resource-group defect-detection-rg \
    --name defectdetectionacr \
    --sku Basic \
    --admin-enabled true
```

---

## 3. Build the Docker image

```bash
docker build -t defectdetectionacr.azurecr.io/defect-api:v1 .
```

---

## 4. Push image to Azure Container Registry

```bash
az acr login --name defectdetectionacr

docker push defectdetectionacr.azurecr.io/defect-api:v1
```

---

## 5. Deploy to Azure Container Instances

```bash
az container create \
    --resource-group defect-detection-rg \
    --name defect-api-instance \
    --image defectdetectionacr.azurecr.io/defect-api:v1 \
    --registry-login-server defectdetectionacr.azurecr.io \
    --registry-username $(az acr credential show -n defectdetectionacr --query username -o tsv) \
    --registry-password $(az acr credential show -n defectdetectionacr --query passwords[0].value -o tsv) \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --dns-name-label defect-detection-api \
    --restart-policy Always
```

---

## 6. Test the deployment

```bash
echo "API available at:"
echo "http://defect-detection-api.westeurope.azurecontainer.io:8000"

curl http://defect-detection-api.westeurope.azurecontainer.io:8000/health
```

---

# Option C: App Service from Docker Container

This approach deploys a Docker image stored in Azure Container Registry.

---

## 1. Create an App Service Plan

```bash
az appservice plan create \
    --resource-group defect-detection-rg \
    --name defect-detection-plan \
    --sku B1 \
    --is-linux
```

---

## 2. Create a Web App from the container image

```bash
az webapp create \
    --resource-group defect-detection-rg \
    --plan defect-detection-plan \
    --name defect-detection-api-app \
    --deployment-container-image-name defectdetectionacr.azurecr.io/defect-api:v1
```

---

## 3. Enable container logging

```bash
az webapp log config \
    --resource-group defect-detection-rg \
    --name defect-detection-api-app \
    --docker-container-logging filesystem
```

---

## 4. Mount persistent Azure Files storage

Create the file share:

```bash
az storage share create \
    --name defect-checkpoints \
    --account-name defectstorage
```

Attach it to the Web App:

```bash
az webapp config storage-account add \
    --resource-group defect-detection-rg \
    --name defect-detection-api-app \
    --custom-id checkpoints \
    --storage-type AzureFiles \
    --share-name defect-checkpoints \
    --account-name defectstorage \
    --access-key <storage-key> \
    --mount-path /app/checkpoints
```

---

# Option D: Azure Kubernetes Service (AKS)

AKS is recommended for production deployments requiring orchestration, scaling, and high availability.

---

## Create an AKS cluster

```bash
az aks create \
    --resource-group defect-detection-rg \
    --name defect-detection-aks \
    --node-count 2 \
    --enable-managed-identity
```

---

## Retrieve Kubernetes credentials

```bash
az aks get-credentials \
    --resource-group defect-detection-rg \
    --name defect-detection-aks
```

---

## Deploy the application

```bash
kubectl apply -f deploy/kubernetes.yaml
```

---

# Production Recommendations

| Aspect | Recommendation |
|--------|----------------|
| CPU | Minimum 2 vCPU |
| Memory | Minimum 4 GB |
| Storage | Azure Files for checkpoints |
| Scaling | Enable App Service auto-scaling |
| Monitoring | Enable Application Insights |
| Networking | Use VNet Integration for internal APIs |
| Estimated Cost | ACI ≈ $30/month, App Service B1 ≈ $45/month |

---

# Resource Cleanup

To avoid ongoing Azure charges, delete the Resource Group after testing:

```bash
az group delete \
    --name defect-detection-rg \
    --yes \
    --no-wait
```

Deleting the resource group removes all associated Azure resources, including:

- App Service
- App Service Plan
- Azure Container Registry
- Azure Container Instances
- AKS Cluster
- Storage Accounts
- Networking resources
````
