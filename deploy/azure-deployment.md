# Wdrożenie Defect Detection API na Microsoft Azure

## Opcje wdrożenia

### 1. Azure Container Instances (ACI) — najprostsze

Wymaga: Docker Desktop, Azure CLI, konto Azure (darmowy tier $200 kredytu)

```bash
# 1. Zaloguj się do Azure
az login

# 2. Utwórz Azure Container Registry
az acr create --resource-group defect-detection-rg \
    --name defectdetectionacr --sku Basic --admin-enabled true

# 3. Zbuduj obraz lokalnie
docker build -t defectdetectionacr.azurecr.io/defect-api:v1 .

# 4. Push do ACR
az acr login --name defectdetectionacr
docker push defectdetectionacr.azurecr.io/defect-api:v1

# 5. Uruchom na ACI
az container create \
    --resource-group defect-detection-rg \
    --name defect-api-instance \
    --image defectdetectionacr.azurecr.io/defect-api:v1 \
    --registry-login-server defectdetectionacr.azurecr.io \
    --registry-username $(az acr credential show -n defectdetectionacr --query username -o tsv) \
    --registry-password $(az acr credential show -n defectdetectionacr --query passwords[0].value -o tsv) \
    --cpu 2 --memory 4 \
    --ports 8000 \
    --dns-name-label defect-detection-api \
    --restart-policy Always

# 6. Sprawdź
echo "API dostępne pod: http://defect-detection-api.westeurope.azurecontainer.io:8000"
curl http://defect-detection-api.westeurope.azurecontainer.io:8000/health
```

### 2. Azure App Service (Web App for Containers)

```bash
# 1. Utwórz App Service Plan
az appservice plan create \
    --resource-group defect-detection-rg \
    --name defect-detection-plan \
    --sku B1 \
    --is-linux

# 2. Utwórz Web App
az webapp create \
    --resource-group defect-detection-rg \
    --plan defect-detection-plan \
    --name defect-detection-api-app \
    --deployment-container-image-name defectdetectionacr.azurecr.io/defect-api:v1

# 3. Włącz logowanie
az webapp log config \
    --resource-group defect-detection-rg \
    --name defect-detection-api-app \
    --docker-container-logging filesystem

# 4. Skonfiguruj stałe dane
# Dla Azure Files (przechowywanie checkpointów):
az storage share create --name defect-checkpoints --account-name defectstorage

# Podłącz do App Service
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

### 3. Azure Kubernetes Service (AKS) — skalowalne

```bash
# Utwórz klaster AKS
az aks create \
    --resource-group defect-detection-rg \
    --name defect-detection-aks \
    --node-count 2 \
    --enable-managed-identity

# Pobierz kubeconfig
az aks get-credentials \
    --resource-group defect-detection-rg \
    --name defect-detection-aks

# Wdróż
kubectl apply -f deploy/kubernetes.yaml
```

### Testowanie wdrożenia

```bash
# Health check
curl http://<your-url>/health

# Predykcja
curl -X POST http://<your-url>/predict \
    -F "file=@test_image.jpg" \
    | jq .

# Heatmap
curl -X POST http://<your-url>/predict/with-heatmap-image \
    -F "file=@test_image.jpg" \
    --output prediction_with_heatmap.jpg
```

## Uwagi produkcyjne

| Aspekt | Zalecenie |
|--------|-----------|
| **CPU** | Min 2 vCPU (PyTorch wymaga) |
| **RAM** | Min 4 GB |
| **Storage** | Użyj Azure Files dla checkpointów |
| **Skalowanie** | App Service: włącz auto-scaling |
| **Monitoring** | Application Insights |
| **Koszt** | ACI ~$30/mies, App Service B1 ~$45/mies |

## Czyszczenie zasobów

```bash
az group delete --name defect-detection-rg --yes --no-wait
