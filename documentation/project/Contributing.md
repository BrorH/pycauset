# Contributing to PyCauset

Thank you for your interest in contributing to PyCauset! We welcome contributions from the community, whether it's fixing bugs, adding new features, or improving documentation.

Before you start, read the project protocols:

- [[project/protocols/index|project/protocols/]]
- [[project/protocols/Documentation Protocol|Documentation Protocol]]

## Getting Started

### Prerequisites

To build PyCauset from source, you will need:

*   **Python 3.10+** and **NumPy 2** (CI tests Python 3.12)
*   **C++ Compiler**:
    *   **Windows**: Visual Studio 2022 (Desktop development with C++)
    *   **Linux**: GCC or Clang
    *   **macOS**: Xcode Command Line Tools
*   **CMake 3.24+**
*   **OpenBLAS and LAPACKE** (Linux: `libopenblas-dev liblapacke-dev`; the wheels bundle them)

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

3.  **Install from source** (this builds the native extension via `scikit-build-core`):
    ```bash
    pip install -e .
    ```

## Building the Project

`pip install -e .` is the build on every platform. On Windows, `./tools/build.ps1` wraps
it (choose the Python with `-PythonVersion`). For quick rebuilds of the extension alone,
see [[dev/Build System|Build System]].

## Running Tests

PyCauset has a test suite covering both the C++ core and the Python interface.

### Python Tests
```bash
pytest                          # tests/engine and tests/blockmatrix
PYCAUSET_LAZY=0 pytest tests/engine
pytest tests/atlas              # every op x dtype x structure x view against NumPy
```

### C++ Unit Tests
Configure with `-DBUILD_TESTS=ON`, then `ctest --test-dir <build dir>`.

See [[dev/Testing & Benchmarks|Testing & Benchmarks]].

## Coding Standards

### C++
*   **C++20**.
*   Follow standard RAII principles.
*   Use `pybind11` for Python bindings.
*   Keep headers in `include/` and implementation in `src/`.

### Python
*   Follow **PEP 8**.
*   Use type hints.
*   Every public function and class has a page under `documentation/docs/` (a test checks).

## Documentation

Documentation is written in Markdown and built with MkDocs.

Install documentation dependencies:

```bash
pip install -r requirements-docs.txt
```

If you are changing behavior, remember the docs checklist in [[project/protocols/Documentation Protocol|Documentation Protocol]].

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
