# Build System (Canonical Workflow)

This page documents **how PyCauset is built** and where build configuration lives.

## Canonical build: pip + scikit-build-core

PyCauset’s canonical build path is Python packaging via `pyproject.toml` using **scikit-build-core**, which drives **CMake**.

That means:
- The “real” build configuration lives in `CMakeLists.txt`.
- Running `pip install .` or `pip install -e .` will configure and build the C++ extension.

## Why this matters

- It prevents having multiple divergent build systems.
- It matches how users install from PyPI (prebuilt wheels or source builds).
- It reduces version/installation confusion by making “pip build” the source of truth.

## Where compiler flags and warnings live

- **Compiler flags / warning suppressions / link settings** live in `CMakeLists.txt`.
- When we migrate scripts (e.g., `build.ps1`) to be pip wrappers, **we are not losing these flags**: pip → scikit-build-core → CMake uses them.

## Developer workflows

### Editable install (recommended for development)

- `pip install -e .`

This builds the native extension and installs the Python package in editable mode.

### Non-editable build (local install)

- `pip install .`

### Passing CMake options

scikit-build-core supports passing CMake configuration through environment variables.

Common patterns:

- Set CMake arguments:
  - `CMAKE_ARGS="-DENABLE_CUDA=ON" pip install -e .`

- Choose build type (platform-dependent):
  - On Windows/MSVC, CMake uses multi-config generators; Release/Debug are selected at build time.
  - On single-config generators (many Linux setups), you pass `-DCMAKE_BUILD_TYPE=Release`.

(Exact invocation details may vary by OS/toolchain; keep this page updated as we standardize.)

## Wrapper scripts policy

We may keep convenience scripts like `build.ps1`, but **they must remain wrappers** around the canonical pip-based commands.

- Allowed: “one-liner wrappers” calling pip with standard args.
- Not allowed: scripts that introduce separate build flags, separate output layouts, or separate dependency logic.

## build.ps1

`build.ps1` is intentionally small and only wraps `pip`.

- Editable install (default): `./build.ps1`
- Non-editable install: `./build.ps1 -Action install`
- Build a wheel into `dist/`: `./build.ps1 -Action wheel`

### Choosing a Python

- Resolve via Windows py launcher: `./build.ps1 -PythonVersion 3.12`
- Use an explicit interpreter: `./build.ps1 -PythonExe C:\\Path\\To\\python.exe`

### Passing CMake arguments

Use `-CMakeArg` to add flags for this run (it appends to `CMAKE_ARGS` for the pip build):

- `./build.ps1 -CMakeArg "-DENABLE_CUDA=ON"`
- `./build.ps1 -CMakeArg "-DCMAKE_BUILD_TYPE=Release"`
