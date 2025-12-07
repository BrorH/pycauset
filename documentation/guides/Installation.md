# Installation Guide

## Method 1: Install via pip (Recommended)
The easiest way to install PyCauset is using `pip`. 

### From PyPI
To install the latest published version:
```bash
pip install pycauset
```

## Method 2: Building from Source
If you want to build from source or contribute to development, you will need a C++ compiler.

### Prerequisites for Source Build
*   **Windows**: Install **Visual Studio Community 2022** (or Build Tools) with the **"Desktop development with C++"** workload.
*   **Linux**: Install `g++` or `clang` (e.g., `sudo apt install build-essential`).
*   **macOS**: Install Xcode Command Line Tools (`xcode-select --install`).

### Install from Source
To install from the local repository (this will compile the C++ extension):

```bash
pip install .
```

To install in editable mode (for development):

```bash
pip install -e .
```

## Method 3: Manual Build (Development)
If you are developing the C++ core or prefer manual control over the build process, you can use the provided PowerShell script (`build.ps1`). This script handles CMake configuration, compilation, and testing.

### 1. Verify Compiler
Ensure your compiler is discoverable in your terminal.
*   **Windows**: Run `cl`. If it's not found, you may need to launch the "Developer PowerShell for VS 2022" or add MSVC to your PATH.

### 2. Build Script
Use the `build.ps1` script to build the project:

```powershell
# Build everything (C++ tests + Python module)
./build.ps1 -All

# Only build the Python module (copies to python/pycauset)
./build.ps1 -Python

# Only run C++ unit tests
./build.ps1 -Tests
```

### 3. Running Scripts
If you used `./build.ps1 -Python` (and didn't use `pip`), the compiled module is located in `python/pycauset`. You can run scripts by ensuring this directory is in your `PYTHONPATH` or by running scripts from the root directory.

## GPU Acceleration Troubleshooting

If `pycauset.cuda.is_available()` returns `False` even though you have an NVIDIA GPU, follow these steps.

### 1. The Problem: Missing CUDA Toolkit
PyCauset needs to **compile** custom CUDA code (`.cu` files) to run on your GPU. This requires the **NVIDIA CUDA Toolkit**, which is different from just having the NVIDIA Drivers installed.

The build system checks for the `nvcc` compiler. If it can't find it, it silently skips building the GPU backend.

### 2. How to Fix

#### Step 1: Install CUDA Toolkit
1.  Go to the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page.
2.  Select **Windows** -> **x86_64** -> **10** (or 11) -> **exe (local)**.
3.  Download and run the installer.
4.  **Important**: During installation, choose "Express" or ensure "Development" components are selected.

#### Step 2: Verify Installation
Open a **new** PowerShell terminal (to refresh environment variables) and run:
```powershell
nvcc --version
```
You should see output like `Cuda compilation tools, release 12.x...`.

#### Step 3: Rebuild PyCauset
Once `nvcc` is working, rebuild the project:
```powershell
.\build.ps1 -Python
```
Watch the output for:
```
-- Found CUDA: ...
-- Building pycauset_cuda accelerator.
```

#### Step 4: Verify in Python
```python
import pycauset
print(pycauset.cuda.is_available())  # Should be True
print(pycauset.cuda.current_device()) # Should be your GPU name
```

### Common Issues
*   **"nvcc not found" after install**: You may need to manually add the CUDA `bin` directory to your PATH.
    *   Default: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
*   **Visual Studio Integration**: CUDA requires a C++ compiler (MSVC). Ensure you have Visual Studio (Community Edition is fine) installed with "Desktop development with C++".
