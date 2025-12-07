# Contributing to PyCauset

Thank you for your interest in contributing to PyCauset! We welcome contributions from the community, whether it's fixing bugs, adding new features, or improving documentation.

## Getting Started

### Prerequisites

To build PyCauset from source, you will need:

*   **Python 3.8+**
*   **C++ Compiler**:
    *   **Windows**: Visual Studio 2022 (Desktop development with C++)
    *   **Linux**: GCC or Clang
    *   **macOS**: Xcode Command Line Tools
*   **CMake 3.15+**
*   **CUDA Toolkit** (Optional, for GPU support)

### Setting Up the Development Environment

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/BrorH/pycauset.git
    cd pycauset
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\Activate.ps1
    # Linux/macOS
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Building the Project

We provide a `build.ps1` script (Windows) and `build.sh` (Linux/macOS) to simplify the build process.

### Windows

```powershell
# Build Python extension and run tests
./build.ps1 -All

# Only build Python extension
./build.ps1 -Python
```

### Linux/macOS

```bash
# Build Python extension
./build.sh
```

## Running Tests

PyCauset has a comprehensive test suite covering both the C++ core and the Python interface.

### Python Tests
```bash
pytest tests/python
```

### C++ Unit Tests
The C++ tests are compiled into an executable.
```powershell
# Windows
./build/tests/Release/causal_tests.exe
```

## Coding Standards

### C++
*   Use **C++17** features.
*   Follow standard RAII principles.
*   Use `pybind11` for Python bindings.
*   Keep headers in `include/` and implementation in `src/`.

### Python
*   Follow **PEP 8**.
*   Use type hints.
*   Document all public functions and classes using Google-style docstrings.

## Documentation

Documentation is written in Markdown and built with MkDocs.

```bash
# Serve documentation locally
mkdocs serve
```

## Submitting a Pull Request

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/my-feature`).
3.  Commit your changes.
4.  Push to the branch (`git push origin feature/my-feature`).
5.  Open a Pull Request.

Please ensure all tests pass before submitting!
