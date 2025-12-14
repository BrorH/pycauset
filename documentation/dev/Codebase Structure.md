# Codebase Structure (Overview)

This page explains **where things live** in the repository and **how the pieces fit together**.

## Core philosophy (non-negotiable)

- **PyCauset is “NumPy for causal sets”.** Users interact with **top-level** Python APIs: `pycauset.Matrix`, `pycauset.CausalMatrix`, `pycauset.matmul`, etc.
- Storage, dtype selection, backend dispatch (CPU/GPU), and performance optimizations are **automatic and behind the scenes**.
- Internal code organization may change, but **the public surface stays stable**.

## Repository map

### Python package (user-facing)

- `python/pycauset/`
  - Public Python API and higher-level convenience wrappers.
  - The native extension is imported as `pycauset._pycauset`.

### C++ core (engine)

- `include/pycauset/` — public C++ headers (engine API)
- `src/` — C++ implementations
  - `src/core/` — memory mapping, storage utils, I/O accelerator, system utils
  - `src/matrix/` — matrix types (dense, triangular, bit, etc.)
  - `src/vector/` — vector types
  - `src/compute/` — compute dispatch architecture
    - `ComputeContext` (singleton)
    - `AutoSolver` (routes CPU vs GPU)
    - `cpu/` (CPU device + solvers)
  - `src/accelerators/cuda/` — optional CUDA plugin (loaded dynamically)
  - `src/bindings.cpp` — Python bindings (pybind11)

### Tests & benchmarks

- `tests/python/` — Python interface + integration tests
- `tests/*.cpp` — C++ unit tests (engine-level)
- `benchmarks/` — performance scripts and comparison suites

### Build system

- `pyproject.toml` — canonical Python build entry (scikit-build-core)
- `CMakeLists.txt` — C++ build configuration and compiler flags
- `build.ps1` — (planned) thin wrapper around the canonical pip build commands

## How a user call flows through the stack

### Example: `C = A @ B` (matrix multiplication)

1. User code calls `A @ B` (Python operator) or `pycauset.matmul(A, B)`.
2. Python calls into the native extension (`pycauset._pycauset`) or a thin Python wrapper.
3. Native code performs:
   - type resolution / allocation,
   - device dispatch (CPU vs GPU) via `ComputeContext` + `AutoSolver`,
   - algorithm selection (direct vs streaming) via `MemoryGovernor` and solver logic.
4. The result is returned as a Python object that wraps a C++ `MatrixBase` implementation.

### Example: out-of-core optimization

When a matrix is disk-backed (memory-mapped), solvers can:
- emit access-pattern hints (sequential/strided),
- trigger prefetching via the I/O accelerator,
- avoid page-fault thrashing during streaming kernels.

## Ownership rules (where new code should go)

### If you add/change a user-facing Python API
- Add docs under `documentation/docs/` + `documentation/guides/`.
- Keep the *public* entrypoint top-level (`pycauset.*`).
- Prefer a thin wrapper that delegates to the C++ engine.

### If you add/change an engine feature
- Update the relevant C++ subsystem:
  - storage/memory: `src/core/`, `include/pycauset/core/`
  - matrix/vector types: `src/matrix/`, `src/vector/`
  - compute/dispatch: `src/compute/`
  - CUDA: `src/accelerators/cuda/`
- Update internals docs under `documentation/internals/`.

### If you add/change bindings
- Update bindings code and ensure tests cover the Python-level behavior.
- Add/update the binding checklist in [[dev/Bindings & Dispatch]].

## Roadmap constraint: NxM matrices

Today, much of the engine assumes square matrices (NxN). The roadmap includes **NxM support for all matrix types** (dense, triangular, symmetric, bit, etc.), while explicitly avoiding N-D arrays.

When reorganizing or refactoring, avoid hard-coding “square-only” assumptions into new architecture layers.
