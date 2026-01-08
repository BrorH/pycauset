<#
.SYNOPSIS
  Thin build wrapper for PyCauset (pip/scikit-build-core).

.DESCRIPTION
  PyCauset's canonical build path is Python packaging via `pyproject.toml`.
  This script stays intentionally small: it only calls `pip` and passes
  through optional CMake arguments.

  Policy: compiler flags/warning suppressions belong in `CMakeLists.txt`.
  This script must not become a second build system.

.EXAMPLE
  ./build.ps1
  Performs an editable install (`pip install -e .`).

.EXAMPLE
  ./build.ps1 -PythonVersion 3.12 -CMakeArg "-DENABLE_CUDA=ON"

.EXAMPLE
  ./build.ps1 -Action wheel -CMakeArg "-DCMAKE_BUILD_TYPE=Release"
#>

[CmdletBinding()]
param(
    [ValidateSet('editable', 'install', 'wheel')]
    [string]$Action = 'editable',

    [string]$PythonVersion,
    [string]$PythonExe,

    [string[]]$CMakeArg = @(),
    [string[]]$PipArg = @(),

    [switch]$NoBuildIsolation,
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-PythonExecutable {
    param(
        [string]$PythonVersion,
        [string]$PythonExe
    )

    if ($PythonExe) {
        if (-not (Test-Path $PythonExe)) {
            throw "Provided PythonExe path '$PythonExe' was not found."
        }
        return (Resolve-Path $PythonExe).Path
    }

    if ($PythonVersion) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw "PythonVersion was provided but the 'py' launcher was not found. Install Python for Windows or pass -PythonExe."
        }

        $resolved = (& py "-$PythonVersion" -c "import sys; print(sys.executable)" 2>$null).Trim()
        if (-not $resolved) {
            throw "Unable to resolve Python $PythonVersion via py launcher."
        }
        if (-not (Test-Path $resolved)) {
            throw "Resolved Python path '$resolved' does not exist."
        }
        return $resolved
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "No Python found on PATH. Install Python or pass -PythonExe/-PythonVersion."
    }
    return $cmd.Source
}

function Remove-LocalBuildArtifacts {
    $paths = @(
        'build',
        'dist',
        '.pycauset',
        '_skbuild',
        '.pytest_cache'
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
        }
    }
}

$python = Resolve-PythonExecutable -PythonVersion $PythonVersion -PythonExe $PythonExe
Write-Host "Using Python: $python" -ForegroundColor Yellow

if ($Clean) {
    Write-Host "Cleaning local build artifacts..." -ForegroundColor Cyan
    Remove-LocalBuildArtifacts
}

$oldCmakeArgs = $env:CMAKE_ARGS
try {
    if ($CMakeArg.Count -gt 0) {
        $extra = ($CMakeArg -join ' ')
        if ($env:CMAKE_ARGS) {
            $env:CMAKE_ARGS = ($env:CMAKE_ARGS + ' ' + $extra)
        } else {
            $env:CMAKE_ARGS = $extra
        }
        Write-Host "CMAKE_ARGS=$($env:CMAKE_ARGS)" -ForegroundColor DarkGray
    }

    $pipBase = @('-m', 'pip')
    if ($NoBuildIsolation) {
        $PipArg = @('--no-build-isolation') + $PipArg
    }

    switch ($Action) {
        'editable' {
            Write-Host "Running: pip install -e ." -ForegroundColor Cyan
            & $python @pipBase install -e . @PipArg
        }
        'install' {
            Write-Host "Running: pip install ." -ForegroundColor Cyan
            & $python @pipBase install . @PipArg
        }
        'wheel' {
            Write-Host "Running: pip wheel ." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path dist | Out-Null
            & $python @pipBase wheel . -w dist @PipArg
        }
        default {
            throw "Unknown action: $Action"
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "pip exited with code $LASTEXITCODE"
    }
} finally {
    $env:CMAKE_ARGS = $oldCmakeArgs
}

Write-Host "Done." -ForegroundColor Green
