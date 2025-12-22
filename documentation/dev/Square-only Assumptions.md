# Square-only assumptions (NxN)

PyCauset supports rectangular dense matrices (rows×cols) in the core engine, including bit-packed boolean matrices. This document tracks what is *still* square-only, either by mathematical definition or by implementation constraints.

## C++ engine: remaining square-only areas

### Square-only types (by definition)

Some matrix families are inherently square:

- Triangular matrices (including causal matrices)
- Symmetric / antisymmetric matrices
- Diagonal matrices

Identity matrices are **shape-flexible** in PyCauset: `IdentityMatrix(rows, cols)` (and `pycauset.identity([rows, cols])`) creates an identity-like matrix with ones on the diagonal up to `min(rows, cols)`.

The factory enforces this at creation time: these `MatrixType`s reject `rows != cols` (triangular/causal, diagonal, symmetric/antisymmetric).

### DenseBitMatrix is bit-packed (implementation detail)

`DenseBitMatrix` (`DenseMatrix<bool>`) is stored in bit-packed row layout with a per-row stride determined by `cols` (not by `n`). Rectangular `(rows, cols)` shapes are supported.

### Square-only operations

Some operations require square matrices even for dense numeric types:

- Determinant
- Inverse

These should fail fast with a clear error when `rows != cols`.

### Optional algorithms / build-dependent APIs

Some higher-level solvers (for example, eigensolvers) are build-dependent. Tests and docs should treat these as optional and skip/disable features when the bindings are not present.

### Triangular types are inherently square

- Triangular matrices (and causal matrices) are square by definition.

Implication: NxM support does **not** apply to triangular/causal matrices as a general concept; NxM primarily targets dense/symmetric/rectangular operations.

## What is no longer square-only (Phase 1)

- `MatrixBase` / `PersistentObject` track `rows` and `cols`; `size()` is total elements (`rows * cols`).
- Dense numeric matrices support rectangular storage and indexing.
- `ObjectFactory::create_matrix(rows, cols, ...)` exists and is used in core math paths.
- Persistence stores `rows`/`cols` in typed metadata and loaders prefer rectangular constructors.
- Python allocation (`zeros` / `ones` / `empty`) supports rectangular shapes for dense numeric dtypes.

## Python layer: remaining square-only surfaces

- Triangular/causal matrix constructors are square-only by definition.

## See also

- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
- [[internals/Memory and Data|internals/Memory and Data]]
- [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]

## Bindings

- The pybind `shape` property is derived from `rows()` and `cols()`, so rectangular dense numeric matrices surface correct `(rows, cols)` shapes in Python.
