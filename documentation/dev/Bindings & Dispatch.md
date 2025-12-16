# Bindings & Dispatch (Python ↔ C++ Engine)

This page explains how Python calls reach the C++ engine, and how CPU/GPU dispatch works.

## High-level architecture

- **Python public API** lives at `pycauset.*`.
- **Native extension module** is imported as `pycauset._pycauset`.
- **C++ engine** provides:
  - storage/memory mapping,
  - matrix/vector types,
  - compute dispatch (CPU/GPU),
  - solvers (BLAS/LAPACK, streaming kernels, CUDA plugin).

## Dispatch path (device selection)

- `ComputeContext` owns the active device configuration.
- `AutoSolver` routes operations:
  - CPU for small problems or unsupported types,
  - GPU for large problems when available and supported.
- “Direct vs streaming” selection is coordinated by memory/governor logic and solvers.

## Why bindings are a high-risk zone

Bindings are where drift happens:
- Python may assume a class/function exists in native code.
- Native bindings may expose a different name or only a subset.

**Rule:** if Python expects a native symbol, it must be guaranteed (or Python must provide a deliberate fallback with tests).

## Binding checklist (when adding a new feature)

When adding a new matrix type, dtype, or operation:

1. **C++ engine:** implement or route the capability.
2. **Bindings:** expose the capability to Python.
3. **Python surface:** ensure it is reachable via a top-level API (`pycauset.*`).
4. **Tests:** add/extend tests under `tests/python/`.
5. **Docs:** update `documentation/docs/`, `documentation/guides/`, and (if architectural) `documentation/internals/`.

## Bindings layout (modular translation units)

The native extension is built as one module, but the binding code is intentionally split across multiple translation units:

- `src/bindings.cpp` is a thin `PYBIND11_MODULE` entrypoint.
- `src/bindings/` contains the binding implementations, grouped by subsystem.
  - Examples include: `bind_core.cpp`, `bind_matrix.cpp`, `bind_vector.cpp`, `bind_causet.cpp`, `bind_complex.cpp`.

When adding a new bound type or function:

- Put the binding code in the most appropriate `src/bindings/bind_*.cpp` file (or add a new one if needed).
- Ensure the symbol is exported under the expected name in `pycauset._pycauset`.
- Add/extend a Python test that imports it through the intended public surface (usually `pycauset.*`).

## Drift check (native exports)

Bindings refactors can silently drop exports that Python relies on.

Run the repo-level drift check from the project root:

- `python tools/check_native_exports.py`

It imports the in-tree Python package (`python/pycauset`) and asserts that `pycauset._pycauset` exports the required symbols.

Options:

- `python tools/check_native_exports.py --no-smoke` (presence-only)
- `python tools/check_native_exports.py --strict` (fail on optional exports too)

## Naming rule (public API stability)

Internal modules/folders may be reorganized. Prefer keeping the user entrypoints stable and top-level:
- `pycauset.Matrix`
- `pycauset.CausalMatrix`
- `pycauset.matmul`
- `pycauset.inverse` / `~M`
- `MatrixBase.trace` / `MatrixBase.determinant`

Pre-alpha note: the surface *may* change if it improves the architecture, but treat it as a major design decision: get explicit approval and update Philosophy + Protocols + tests.
