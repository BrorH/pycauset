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
*   Python 3.10+ and NumPy 2; CMake 3.24+; OpenBLAS and LAPACKE (Linux: `sudo apt install libopenblas-dev liblapacke-dev`).
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

## Method 3: Rebuilding the extension during development

For C++ work, rebuild only the extension and run from the source tree:

```bash
cmake --build build/cpp --target _pycauset
cp build/cpp/_pycauset*.so build/cpp/libpycauset_core.so python/pycauset/
PYTHONPATH=python python -m pytest
```

On Windows, `./tools/build.ps1` wraps the pip build (`-Action editable|install|wheel`,
`-PythonVersion 3.12`). Details: [[dev/Build System|Build System]].
