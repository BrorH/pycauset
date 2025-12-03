param(
    [switch]$Tests,
    [switch]$Python,
    [switch]$All,
    [string]$PythonVersion,
    [string]$PythonExe
)

if (-not ($Tests -or $Python -or $All)) {
    $All = $true
}

if ($All) {
    $Tests = $true
    $Python = $true
}

$resolvedPythonExe = $null
if ($PythonExe) {
    if (Test-Path $PythonExe) {
        $resolvedPythonExe = (Resolve-Path $PythonExe).Path
    } else {
        Write-Host "Provided PythonExe path '$PythonExe' was not found." -ForegroundColor Red
        exit 1
    }
} elseif ($PythonVersion) {
    Write-Host "Resolving Python $PythonVersion using py launcher..." -ForegroundColor Cyan
    try {
        $pyArgs = @("-$PythonVersion", "-c", "import sys; print(sys.executable)")
        $resolvedPythonExe = (& py @pyArgs).Trim()
        if (-not $resolvedPythonExe) {
            throw "py launcher did not return a path."
        }
    } catch {
        Write-Host "Unable to resolve Python version $PythonVersion via py launcher: $_" -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path $resolvedPythonExe)) {
        Write-Host "Resolved Python path '$resolvedPythonExe' does not exist." -ForegroundColor Red
        exit 1
    }
}

if ($resolvedPythonExe) {
    Write-Host "Using Python executable: $resolvedPythonExe" -ForegroundColor Yellow
}

Write-Host "Checking for CMake..."
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $commonPaths = @(
        "C:\Program Files\CMake\bin",
        "C:\Program Files (x86)\CMake\bin"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path "$path\cmake.exe") {
            Write-Host "Found CMake at $path. Adding to session PATH." -ForegroundColor Yellow
            $env:Path = "$path;$env:Path"
            break
        }
    }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "Error: CMake is not in your PATH." -ForegroundColor Red
    exit 1
}

$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    Write-Host "Creating build directory..."
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $buildDir

$cmakeConfigureArgs = @("..")
if ($Tests) {
    $cmakeConfigureArgs += "-DBUILD_TESTS=ON"
}
if ($resolvedPythonExe) {
    $normalizedExe = $resolvedPythonExe -replace "\\", "/"
    $cmakeConfigureArgs = @("-DPython3_EXECUTABLE:FILEPATH=$normalizedExe", "-DPython_EXECUTABLE:FILEPATH=$normalizedExe") + $cmakeConfigureArgs
}

Write-Host "Configuring project..." -ForegroundColor Cyan
cmake @cmakeConfigureArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Configuration failed." -ForegroundColor Red
    Pop-Location
    exit 1
}

if ($Tests) {
    Write-Host "Building tests..." -ForegroundColor Cyan
    cmake --build . --config Release --target causal_tests
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Test build failed." -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Write-Host "Running tests..." -ForegroundColor Cyan
    $possiblePaths = @(
        ".\Release\causal_tests.exe",
        ".\tests\Release\causal_tests.exe",
        ".\causal_tests.exe",
        ".\tests\causal_tests.exe"
    )

    $testExe = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $testExe = $path
            break
        }
    }

    if ($testExe) {
        Write-Host "Found executable at: $testExe" -ForegroundColor Gray
        & $testExe
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Tests failed." -ForegroundColor Red
            Pop-Location
            exit 1
        }
    } else {
        Write-Host "Could not find test executable." -ForegroundColor Red
        Pop-Location
        exit 1
    }
}

if ($Python) {
    Write-Host "Building pycauset module..." -ForegroundColor Cyan
    cmake --build . --config Release --target _pycauset
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python module build failed." -ForegroundColor Red
        Pop-Location
        exit 1
    }

    $pydFile = Get-ChildItem -Recurse -Filter "_pycauset*.pyd" | Select-Object -First 1
    if ($pydFile) {
        $packageDir = Join-Path .. "python/pycauset"
        if (-not (Test-Path $packageDir)) {
            New-Item -ItemType Directory -Path $packageDir | Out-Null
        }
        Copy-Item $pydFile.FullName -Destination $packageDir -Force
        # Copy-Item $pydFile.FullName -Destination ..\ -Force
        Write-Host "Copied module to python/pycauset" -ForegroundColor Green
    } else {
        Write-Host "Could not locate generated _pycauset module." -ForegroundColor Red
        Pop-Location
        exit 1
    }
}

Pop-Location
Write-Host "Build complete." -ForegroundColor Green
