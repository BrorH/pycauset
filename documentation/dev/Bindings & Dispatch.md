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

## Naming rule (public API stability)

Internal modules/folders may be reorganized, but the stable user entrypoints should remain:
- `pycauset.Matrix`
- `pycauset.CausalMatrix`
- `pycauset.matmul`
- `pycauset.inverse` / `~M`
- `pycauset.eigvals` / `pycauset.eig`

To change the stable surface, treat it as a major design decision and update Philosophy + Protocols.
