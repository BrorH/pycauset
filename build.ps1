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

# --- CUDA Compatibility Check ---
Write-Host "Checking GPU Compatibility..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ($gpuInfo) {
        $parts = $gpuInfo -split ","
        $gpuName = $parts[0].Trim()
        $computeCap = [double]$parts[1].Trim()
        Write-Host "  Detected GPU: $gpuName (Compute Capability $computeCap)" -ForegroundColor Gray

        # Check NVCC Version
        if (Get-Command nvcc -ErrorAction SilentlyContinue) {
            $nvccOut = nvcc --version | Select-String "release (\d+\.\d+)"
            if ($nvccOut.Matches.Groups[1].Value) {
                $cudaVer = [double]$nvccOut.Matches.Groups[1].Value
                Write-Host "  Detected CUDA Toolkit: v$cudaVer" -ForegroundColor Gray

                # CRITICAL CHECK: CUDA 13+ drops support for CC < 7.0
                if ($cudaVer -ge 13.0 -and $computeCap -lt 7.0) {
                    Write-Host "`n[WARNING] HARDWARE INCOMPATIBILITY DETECTED" -ForegroundColor Yellow
                    Write-Host "  Your GPU ($gpuName) has Compute Capability $computeCap." -ForegroundColor Yellow
                    Write-Host "  The active CUDA Toolkit (v$cudaVer) requires Compute Capability 7.0 or higher." -ForegroundColor Yellow
                    
                    # Search for a compatible CUDA version (v12.x)
                    Write-Host "  Searching for a compatible CUDA version (v12.x)..." -ForegroundColor Cyan
                    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
                    $compatibleFound = $false
                    
                    if (Test-Path $cudaRoot) {
                        $versions = Get-ChildItem $cudaRoot | Where-Object { $_.Name -like "v12.*" } | Sort-Object Name -Descending
                        if ($versions) {
                            $bestVersion = $versions[0]
                            $compatibleFound = $true
                            Write-Host "  Found compatible CUDA Toolkit: $($bestVersion.Name)" -ForegroundColor Green
                            
                            # Set environment variables for this session
                            $env:CUDA_PATH = $bestVersion.FullName
                            $env:Path = "$($bestVersion.FullName)\bin;" + $env:Path
                            Write-Host "  Switched active CUDA Toolkit to $($bestVersion.Name) for this build." -ForegroundColor Green
                        }
                    }

                    if (-not $compatibleFound) {
                        Write-Host "`n[CRITICAL ERROR] NO COMPATIBLE CUDA TOOLKIT FOUND" -ForegroundColor Red -BackgroundColor Black
                        Write-Host "  You must install CUDA Toolkit 12.x to support this GPU." -ForegroundColor Red
                        
                        $choice = Read-Host "  Do you want to download and install CUDA 12.6 automatically? (Y/N)"
                        if ($choice -eq 'Y' -or $choice -eq 'y') {
                            Write-Host "  Launching CUDA 12 Installer script..." -ForegroundColor Green
                            & .\install_cuda12.ps1
                            Write-Host "  Please restart this terminal after installation completes." -ForegroundColor Yellow
                            exit 1
                        } else {
                            Write-Host "  Aborting build. Please install CUDA 12 manually." -ForegroundColor Red
                            exit 1
                        }
                    }
                }
            }
        }
    }
} catch {
    Write-Host "  Could not query GPU info (nvidia-smi not found or failed). Skipping compatibility check." -ForegroundColor DarkGray
}

# --- CUDA Auto-Installation Logic ---
Write-Host "Checking for CUDA Toolkit..."
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    # Check standard location first
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $foundLocal = $false
    if (Test-Path $cudaRoot) {
        $versions = Get-ChildItem $cudaRoot | Where-Object { $_.Name -like "v*" } | Sort-Object Name -Descending
        if ($versions) {
            $best = $versions[0]
            Write-Host "CUDA Toolkit found at $($best.FullName) (not in PATH)." -ForegroundColor Yellow
            Write-Host "Adding to PATH for this session..." -ForegroundColor Cyan
            $env:Path = "$($best.FullName)\bin;" + $env:Path
            $foundLocal = $true
        }
    }

    if (-not $foundLocal) {
        Write-Host "CUDA Toolkit (nvcc) not found." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "Attempting to install NVIDIA CUDA Toolkit via winget..." -ForegroundColor Cyan
            Write-Host "NOTE: This may trigger a UAC prompt and take several minutes (3GB+ download)." -ForegroundColor Yellow
            try {
                $proc = Start-Process winget -ArgumentList "install -e --id Nvidia.CUDA --silent --accept-source-agreements --accept-package-agreements" -Wait -PassThru
                
                if ($proc.ExitCode -eq 0) {
                    Write-Host "CUDA Toolkit installed successfully." -ForegroundColor Green
                    Write-Host "Refreshing environment variables..." -ForegroundColor Cyan
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                } else {
                    Write-Host "CUDA installation failed with exit code $($proc.ExitCode)." -ForegroundColor Red
                }
            } catch {
                Write-Host "Failed to run winget: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "winget not found. Cannot auto-install CUDA." -ForegroundColor Red
        }
    }
} else {
    Write-Host "CUDA Toolkit found." -ForegroundColor Green
}

# --- Build Environment Setup (Ninja + VS) ---
$useNinja = $false
$vcvarsPath = $null

# Check for Ninja
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $useNinja = $true
} else {
    Write-Host "Ninja not found. Attempting to install via pip..." -ForegroundColor Cyan
    try {
        if ($resolvedPythonExe) {
            & $resolvedPythonExe -m pip install ninja
        } else {
            pip install ninja
        }
        if (Get-Command ninja -ErrorAction SilentlyContinue) {
            $useNinja = $true
            Write-Host "Ninja installed successfully." -ForegroundColor Green
        }
    } catch {
        Write-Host "Failed to install Ninja. Falling back to default generator." -ForegroundColor Yellow
    }
}

# Find Visual Studio environment if using Ninja
if ($useNinja) {
    Write-Host "Locating Visual Studio..."
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) {
            $vcvarsPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvarsPath) {
                Write-Host "Found VS environment at: $vcvarsPath" -ForegroundColor Green
            } else {
                Write-Host "vcvars64.bat not found at expected location." -ForegroundColor Yellow
                $useNinja = $false
            }
        } else {
            Write-Host "No suitable Visual Studio installation found." -ForegroundColor Yellow
            $useNinja = $false
        }
    } else {
        Write-Host "vswhere.exe not found." -ForegroundColor Yellow
        $useNinja = $false
    }
}

$buildDir = "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Construct CMake arguments
$cmakeArgs = @("..", "-Wno-dev")

# Force CUDA 12.6 if available (to support Pascal GPUs)
$cuda12Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (Test-Path $cuda12Path) {
    Write-Host "Forcing usage of CUDA 12.6 at $cuda12Path" -ForegroundColor Cyan
    $cuda12Path = $cuda12Path -replace "\\", "/"
    $cmakeArgs += "-DCUDAToolkit_ROOT:PATH=`"$cuda12Path`""
    $cmakeArgs += "-DCMAKE_CUDA_COMPILER:FILEPATH=`"$cuda12Path/bin/nvcc.exe`""
}

if ($Tests) { $cmakeArgs += "-DBUILD_TESTS=ON" }
if ($resolvedPythonExe) {
    $normalizedExe = $resolvedPythonExe -replace "\\", "/"
    $cmakeArgs += "-DPython3_EXECUTABLE:FILEPATH=$normalizedExe"
    $cmakeArgs += "-DPython_EXECUTABLE:FILEPATH=$normalizedExe"
}

# Add the critical flag for newer VS versions
$cmakeArgs += '-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"'
$env:CUDAFLAGS = "-allow-unsupported-compiler"

if ($useNinja -and $vcvarsPath) {
    Write-Host "Using Ninja generator with VS environment..." -ForegroundColor Cyan
    $cmakeArgs += "-G Ninja"
    $cmakeArgs += "-DCMAKE_BUILD_TYPE=Release"
    
    # Create a batch file to run the build in the correct environment
    $batchContent = "@echo off`n"
    $batchContent += "call `"$vcvarsPath`"`n"
    $batchContent += "cd /d `"$((Resolve-Path $buildDir).Path)`"`n"
    $batchContent += "cmake $($cmakeArgs -join ' ')" + "`n"
    $batchContent += "if %errorlevel% neq 0 exit /b %errorlevel%`n"
    
    if ($Tests) {
        # $batchContent += "cmake --build . --config Release --target causal_tests`n"
        # $batchContent += "if %errorlevel% neq 0 exit /b %errorlevel%`n"
    }
    
    if ($Python) {
        $batchContent += "cmake --build . --config Release --parallel`n"
        $batchContent += "if %errorlevel% neq 0 exit /b %errorlevel%`n"
    }
    
    $batchFile = Join-Path $buildDir "build_wrapper.bat"
    Set-Content -Path $batchFile -Value $batchContent
    
    # Run the batch file directly (avoid Start-Process to ensure signals propagate and it doesn't hang)
    & cmd.exe /c "$batchFile"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Using default CMake generator..." -ForegroundColor Yellow
    Push-Location $buildDir
    cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
    
    if ($Tests) {
        # cmake --build . --config Release --target causal_tests -j 1
        # if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
    }
    
    if ($Python) {
        cmake --build . --config Release --parallel
        if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
    }
    Pop-Location
}

# Post-build steps (running tests, copying artifacts)
if ($Tests) {
    Write-Host "Running tests..." -ForegroundColor Cyan
    $possiblePaths = @(
        "$buildDir\causal_tests.exe",
        "$buildDir\Release\causal_tests.exe",
        "$buildDir\tests\causal_tests.exe"
    )
    $testExe = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) { $testExe = $path; break }
    }
    if ($testExe) { & $testExe }
}

if ($Python) {
    $packageDir = "python/pycauset"
    if (-not (Test-Path $packageDir)) { New-Item -ItemType Directory -Path $packageDir | Out-Null }

    # Handle pycauset_core (Shared Library)
    $coreDll = Get-ChildItem -Path $buildDir -Recurse -Filter "pycauset_core.dll" | Select-Object -First 1
    if ($coreDll) {
        Copy-Item $coreDll.FullName -Destination $packageDir -Force
        Write-Host "Copied core library to $packageDir" -ForegroundColor Green
    } else {
        Write-Host "Could not locate pycauset_core.dll" -ForegroundColor Red
    }

    # Handle _pycauset (Main Module)
    $pydFile = Get-ChildItem -Path $buildDir -Recurse -Filter "_pycauset.pyd" | Select-Object -First 1
    if (-not $pydFile) {
        # Fallback: Look for DLL and rename
        $dllFile = Get-ChildItem -Path $buildDir -Recurse -Filter "_pycauset.dll" | Select-Object -First 1
        if ($dllFile) {
            Write-Host "Found _pycauset.dll, copying to .pyd..." -ForegroundColor Cyan
            $pydPath = Join-Path $packageDir "_pycauset.pyd"
            Copy-Item $dllFile.FullName -Destination $pydPath -Force
            
            # Also copy the .dll itself because pycauset_cuda.dll might depend on it by name
            $dllDest = Join-Path $packageDir "_pycauset.dll"
            Copy-Item $dllFile.FullName -Destination $dllDest -Force
            
            Write-Host "Copied module to $pydPath and $dllDest" -ForegroundColor Green
        } else {
            Write-Host "Could not locate generated _pycauset module (pyd or dll)." -ForegroundColor Red
        }
    } else {
        Copy-Item $pydFile.FullName -Destination $packageDir -Force
        Write-Host "Copied module to $packageDir" -ForegroundColor Green
    }

    # Handle pycauset_cuda (Accelerator)
    $cudaDll = Get-ChildItem -Path $buildDir -Recurse -Filter "pycauset_cuda.dll" | Select-Object -First 1
    if ($cudaDll) {
        Copy-Item $cudaDll.FullName -Destination $packageDir -Force
        Write-Host "Copied CUDA accelerator to $packageDir" -ForegroundColor Green

        # Copy CUDA Runtime DLLs for transparency and ease of use
        if ($env:CUDA_PATH) {
            $cudaBin = Join-Path $env:CUDA_PATH "bin"
            # Check for x64 subdirectory (CUDA 13.0+)
            if (Test-Path (Join-Path $cudaBin "x64")) {
                $cudaBin = Join-Path $cudaBin "x64"
            }
            
            Write-Host "Copying CUDA runtime dependencies from $cudaBin..." -ForegroundColor Cyan
            $libs = @("cublas64_*.dll", "cublasLt64_*.dll", "cudart64_*.dll", "cusolver64_*.dll", "cusparse64_*.dll", "nvJitLink*.dll")
            foreach ($lib in $libs) {
                Get-ChildItem -Path $cudaBin -Filter $lib | ForEach-Object {
                    Copy-Item $_.FullName -Destination $packageDir -Force
                    Write-Host "  Copied $($_.Name)" -ForegroundColor Gray
                }
            }
        }
    }
}

Write-Host "Build complete." -ForegroundColor Green
