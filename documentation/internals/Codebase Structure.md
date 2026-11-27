# Codebase Structure

Where things live and how a call flows through them.

## Rules that shape the layout

- **NumPy for causal sets.** Users call top-level functions and operators (`pc.matrix`,
  `pc.causal_matrix`, `A @ B`, `pc.solve`); storage, dtypes, structure and out-of-core
  execution happen behind them.
- **The engine computes.** Python validates, routes and wraps; every result comes from the
  C++ engine, never from NumPy (NumPy is the test oracle).
- **Pre-alpha:** no backwards compatibility; something removed is removed with its docs
  ([[project/protocols/Pre-alpha Policy|Pre-alpha Policy]]).

## Repository map

| Path | Contents |
|---|---|
| `python/pycauset/__init__.py` | the public facade: every `pc.*` name |
| `python/pycauset/_internal/` | implementation modules behind the facade ([[internals/Python Internals]]) |
| `python/pycauset/causet.py`, `spacetime.py`, `synthetic.py`, `vis.py` | causal sets, the Minkowski diamond, synthetic orders, lightweight plots |
| `include/pycauset/`, `src/` | the C++ engine, one folder per subsystem (below) |
| `src/bindings.cpp`, `src/bindings/` | the pybind11 module `pycauset._pycauset`, one file per subsystem ([[internals/Bindings & Dispatch]]) |
| `tests/engine/` | the engine suite (NumPy oracle) |
| `tests/atlas/` | every op x dtype x structure x view against NumPy; timing harness |
| `tests/blockmatrix/` | block matrices |
| `tests/*.cpp` | C++ unit tests (gtest): ParallelUtils, MemoryGovernor, IOAccelerator, copy-on-write, CPU worker |
| `benchmarks/` | speed scripts (`BENCHMARKS.md` at the root) |
| `demos/` | short runnable examples |
| `documentation/` | this site (mkdocs) |
| `experimental/cuda`, `experimental/physics` | parked code: not built, imported, tested or documented with the package |
| `review/` | the ongoing code review: plan, findings, log, operation atlas |
| `tools/` | build and release scripts, export and API-surface checks |

### Engine subsystems

| Folder | Contents |
|---|---|
| `core/` | storage (`PersistentObject`, `MemoryMapper`, `MemoryGovernor`), `ObjectFactory`, `StructureResolver`, dtype dispatch, the kernel log (`DebugTrace`), `ParallelUtils` |
| `matrix/`, `vector/` | the structure classes: dense, triangular (bit-packed for `bool`), symmetric / antisymmetric, Hermitian / skew-Hermitian, diagonal, identity; dense and unit vectors |
| `math/` | entry functions of the operations (`LinearAlgebra`, `Eigen`, `Reductions`, `Bitwise`, `ScalarOps`, `VectorOps`, `CausalMatrix`: transitive reduction); `ViewRead` (typed reads of views), `RawRows.hpp` (bulk reads of stored values) and `ViewValue.hpp`, the one definition of view semantics |
| `compute/` | `ComputeContext`, `AutoSolver`, `StreamingManager`; `cpu/` holds the kernels ([[internals/Compute Architecture]]) |
| `causet/` | the sprinkler and the native Minkowski diamond |

## How a call flows: `C = A @ B`

1. `A @ B` reaches the binding's one matmul helper (`bind_matrix.cpp`); NumPy operands are
   imported first (`numpy_import.hpp`).
2. The engine entry function resolves the result's structure and dtype, normalises views
   (they are read in place, not copied) and allocates the result through `ObjectFactory`
   (RAM or file-backed by the memory threshold).
3. `ComputeContext` hands the call to `CpuDevice`, which runs the kernel directly or, when
   the operands do not fit in RAM, tile by tile through the `StreamingManager`.
4. The result returns as the Python object of its structure class.

The procedure for adding or changing an operation is
[[project/protocols/Adding Operations|Adding Operations]]; for a structure or dtype,
[[project/protocols/Adding Types|Adding Types]].

## Where new code goes

- **User-facing function:** the entry in `__init__.py` delegating to `_internal/` and the
  engine; a page under `documentation/docs/`, a guide section if it changes a workflow;
  tests in `tests/engine/` and an atlas entry for an operation.
- **Engine operation or kernel:** `src/math/` (entry) and `src/compute/cpu/` (kernel), per
  the protocol; docs under `documentation/internals/`.
- **Binding:** the subsystem's `src/bindings/bind_*.cpp`.
