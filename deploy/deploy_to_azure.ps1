<#
.SYNOPSIS
    One-click deployment of Defect Detection API to Azure App Service.
.DESCRIPTION
    Deploys the API from source code without requiring Docker.
    Creates all necessary Azure resources on the free tier.
.PARAMETER AppName
    Unique name for the Azure Web App (default: defect-detection-api).
.PARAMETER Location
    Azure region (default: westeurope).
.PARAMETER ResourceGroup
    Resource group name (default: defect-detection-rg).
.EXAMPLE
    .\deploy\deploy_to_azure.ps1
.EXAMPLE
    .\deploy\deploy_to_azure.ps1 -AppName my-defect-api -Location northeurope
#>
param(
    [string]$AppName = "defect-detection-api",
    [string]$Location = "westeurope",
    [string]$ResourceGroup = "defect-detection-rg"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=" * 60
Write-Host "Defect Detection API - Azure Deployment"
Write-Host "=" * 60
Write-Host "App:            $AppName"
Write-Host "Location:       $Location"
Write-Host "Resource Group: $ResourceGroup"
Write-Host ""

# ---------------------------------------------------------
# Step 1: Check Azure CLI
# ---------------------------------------------------------
Write-Host "[1/6] Checking Azure CLI..."
$azVersion = az version --query '"azure-cli"' -o tsv 2>$null
if (-not $azVersion) {
    Write-Host "ERROR: Azure CLI not found."
    Write-Host "Install it: winget install -e --id Microsoft.AzureCLI"
    Write-Host "Then restart PowerShell and run this script again."
    exit 1
}
Write-Host "  Azure CLI version: $azVersion"

# ---------------------------------------------------------
# Step 2: Check Azure login
# ---------------------------------------------------------
Write-Host "[2/6] Checking Azure login..."
$account = az account show --query user.name -o tsv 2>$null
if (-not $account) {
    Write-Host "  You are not logged in. Launching device code flow..."
    az login --use-device-code
    $account = az account show --query user.name -o tsv
    if (-not $account) {
        Write-Host "ERROR: Login failed. Try running 'az login' manually."
        exit 1
    }
}
Write-Host "  Logged in as: $account"

# ---------------------------------------------------------
# Step 3: Verify model checkpoint exists
# ---------------------------------------------------------
Write-Host "[3/6] Checking model checkpoint..."
$checkpointPath = Join-Path $ProjectRoot "checkpoints" "best_model.pth"
if (-not (Test-Path $checkpointPath)) {
    Write-Host "WARNING: best_model.pth not found at:"
    Write-Host "  $checkpointPath"
    Write-Host "  The API will start, but predictions will fail."
    Write-Host "  Run training first: python -m train.train"
    Write-Host ""
    $continue = Read-Host "  Continue without model? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}
else {
    $sizeMB = [math]::Round((Get-Item $checkpointPath).Length / 1MB, 1)
    Write-Host "  Found: best_model.pth ($sizeMB MB)"
}

# ---------------------------------------------------------
# Step 4: Create resource group
# ---------------------------------------------------------
Write-Host "[4/6] Creating resource group..."
$rgExists = az group exists --name $ResourceGroup -o tsv
if ($rgExists -eq "false") {
    az group create --name $ResourceGroup --location $Location --output none
    Write-Host "  Created: $ResourceGroup ($Location)"
}
else {
    Write-Host "  Already exists: $ResourceGroup"
}

# ---------------------------------------------------------
# Step 5: Create App Service plan and Web App
# ---------------------------------------------------------
Write-Host "[5/6] Creating App Service plan (free tier)..."
$planName = "defect-detection-plan"
$planExists = az appservice plan show `
    --name $planName `
    --resource-group $ResourceGroup `
    --query name -o tsv 2>$null

if (-not $planExists) {
    az appservice plan create `
        --name $planName `
        --resource-group $ResourceGroup `
        --sku F1 `
        --is-linux `
        --output none
    Write-Host "  Created: $planName (F1)"
}
else {
    Write-Host "  Already exists: $planName"
}

Write-Host "[6/6] Creating and configuring Web App..."
$webAppExists = az webapp show `
    --name $AppName `
    --resource-group $ResourceGroup `
    --query name -o tsv 2>$null

if (-not $webAppExists) {
    az webapp create `
        --name $AppName `
        --resource-group $ResourceGroup `
        --plan $planName `
        --runtime "PYTHON:3.11" `
        --output none
    Write-Host "  Created: $AppName"
}
else {
    Write-Host "  Already exists: $AppName"
}

# Configure startup and environment
Write-Host "  Configuring startup script and environment variables..."
az webapp config set `
    --name $AppName `
    --resource-group $ResourceGroup `
    --startup-file "startup.sh" `
    --always-on false `
    --output none

az webapp config appsettings set `
    --name $AppName `
    --resource-group $ResourceGroup `
    --settings `
        PYTHONPATH="/home/site/wwwroot" `
        DEVICE="cpu" `
        PYTHONUNBUFFERED="1" `
    --output none

# ---------------------------------------------------------
# Deploy code
# ---------------------------------------------------------
Write-Host ""
Write-Host "Deploying code (this may take 3-5 minutes on first run)..."
Write-Host ""

Push-Location $ProjectRoot
try {
    az webapp up `
        --name $AppName `
        --resource-group $ResourceGroup `
        --runtime "PYTHON:3.11" `
        --os-type linux
}
finally {
    Pop-Location
}

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
Write-Host ""
Write-Host "=" * 60
Write-Host "DEPLOYMENT COMPLETE"
Write-Host "=" * 60
Write-Host ""
Write-Host "API base URL: https://$AppName.azurewebsites.net"
Write-Host ""
Write-Host "Endpoints:"
Write-Host "  Health check:  https://$AppName.azurewebsites.net/health"
Write-Host "  Swagger docs:  https://$AppName.azurewebsites.net/docs"
Write-Host "  List classes:  https://$AppName.azurewebsites.net/classes"
Write-Host "  Predict:       https://$AppName.azurewebsites.net/predict"
Write-Host ""
Write-Host "Quick smoke test:"
Write-Host "  curl https://$AppName.azurewebsites.net/health"
Write-Host ""
Write-Host "Cleanup (removes all resources):"
Write-Host "  az group delete --name $ResourceGroup --yes --no-wait"
Write-Host ""