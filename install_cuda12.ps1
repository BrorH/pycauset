# install_cuda12.ps1
# Automates the installation of CUDA 12.6 using Windows Package Manager (winget)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "      CUDA 12.6 Automatic Installer (via winget)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check for winget
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'winget' is not found. Please install App Installer from the Microsoft Store." -ForegroundColor Red
    exit 1
}

Write-Host "Installing NVIDIA CUDA Toolkit v12.6..." -ForegroundColor Yellow
Write-Host "This may trigger a UAC prompt. Please accept it." -ForegroundColor Yellow

try {
    # -e: Exact match
    # --id: Nvidia.CUDA
    # --version: 12.6
    # --silent: Request silent installation
    # --accept-*-agreements: Bypass prompts
    winget install --id Nvidia.CUDA --version 12.6 -e --source winget --accept-source-agreements --accept-package-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installation successful!" -ForegroundColor Green
        Write-Host "NOTE: You MUST restart your terminal (or VS Code) for the new environment variables to take effect." -ForegroundColor Magenta
    } else {
        Write-Host "Installation exited with code $LASTEXITCODE." -ForegroundColor Red
    }
} catch {
    Write-Host "Installation failed: $_" -ForegroundColor Red
    exit 1
}

