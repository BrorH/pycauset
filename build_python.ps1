# Build Python Module
Write-Host "Building Python Module..." -ForegroundColor Cyan

# Create build directory
if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }
Push-Location build

# Configure & Build
cmake ..
if ($LASTEXITCODE -ne 0) { Write-Host "Configuration failed." -ForegroundColor Red; Pop-Location; exit 1 }

cmake --build . --config Release --target pycauset
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; Pop-Location; exit 1 }

# Find the generated .pyd file
$pydFile = Get-ChildItem -Recurse -Filter "pycauset*.pyd" | Select-Object -First 1

if ($pydFile) {
    Write-Host "Found module: $($pydFile.FullName)" -ForegroundColor Green
    $packageDir = Join-Path .. "python/pycauset"
    if (-not (Test-Path $packageDir)) {
        New-Item -ItemType Directory -Path $packageDir | Out-Null
    }
    Copy-Item $pydFile.FullName -Destination $packageDir -Force
    Copy-Item $pydFile.FullName -Destination ..\ -Force
    Write-Host "Copied to python/pycauset/ and project root."
} else {
    Write-Host "Could not find generated .pyd file." -ForegroundColor Red
}

Pop-Location
