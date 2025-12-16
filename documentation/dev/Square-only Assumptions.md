# Square-only assumptions (NxN) — Phase F groundwork

PyCauset currently assumes matrices are square (NxN) in many places. This document is a **map of square-only assumptions** so NxM work can be done systematically.

## C++ engine: where “square” is baked in

### MatrixBase is fundamentally square

- `MatrixBase` stores a single `n_` and reports:
  - `rows() == n_`
  - `cols() == n_`
- Many algorithms use `size()` as “dimension”, not “row count”.

Implication: supporting NxM will likely require `rows_` + `cols_` in the base type (and careful handling for transposed views and metadata).

### ObjectFactory uses `n` not `(rows, cols)`

- `ObjectFactory::create_matrix(...)` takes a single `uint64_t n`.
- Many matrix constructors are `DenseMatrix<T>(uint64_t n, ...)`.

Implication: NxM needs new creation APIs (or overloads) that accept `(rows, cols)` and are used consistently across the engine.

### Storage initialization is called with `(n, n)`

- `PersistentObject::initialize_storage(...)` already accepts `rows` and `cols`.
- Most callers pass `n, n`.

Implication: storage metadata can represent NxM, but the type system and constructors currently can’t.

### DenseMatrix assumes `n*n` storage

- `DenseMatrix<T>` allocates `n * n * sizeof(T)` bytes.
- Accessors use `i*n + j` indexing.

Implication: NxM requires storing `rows` and `cols` and indexing with `i*cols + j`.

### Triangular types are inherently square

- Triangular matrices (and causal matrices) are square by definition.

Implication: NxM support does **not** apply to triangular/causal matrices as a general concept; NxM primarily targets dense/symmetric/rectangular operations.

## C++ solvers & math: square-oriented algorithms

- `Eigen` routines typically assume square matrices for eigenvalues/eigenvectors.
- `LinearAlgebra` helpers and many operations are expressed in terms of `size()`.

Implication: NxM work should separate:
- truly square-only operations (det, inverse)
- general NxM operations (matmul, add/sub, elementwise)

## Python layer: current expectations

### `pycauset.Matrix(...)` expects square

- Nested lists/NumPy arrays are treated as square matrices.
- `Matrix(n)` creates an NxN matrix.

### Fallback `pycauset.matmul` is square-limited

- The generic Python fallback currently refuses non-square result matrices.

Implication: once native NxM exists, the Python fallback should either support NxM too or very explicitly raise a helpful error.

## Bindings: current public shape

- The pybind `shape` property is derived from `rows()` and `cols()`.
- Since both equal `n_`, Python sees NxN for all current matrix classes.

## Suggested NxM migration order (docs-only guidance)

1. Introduce a rectangular base matrix type (or evolve `MatrixBase`) to track `(rows, cols)`.
2. Implement `DenseMatrix<T>(rows, cols, ...)` with correct indexing.
3. Extend `ObjectFactory` to create/load/clone `(rows, cols)`.
4. Update storage metadata paths to persist `(rows, cols)`.
5. Update bindings and Python factories to surface NxM for dense matrices.
6. Audit solvers to ensure square-only ops fail fast with clear errors.
