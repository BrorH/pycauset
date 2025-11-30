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

## Method 2: Manual Build (Development)
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
