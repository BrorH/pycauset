# NxM Support Status

PyCauset is actively removing square-only assumptions.

This page is the **discoverable status list** of which operations are currently:

- **NxM-enabled** (rectangular inputs supported), vs
- **restricted / square-only** (by mathematical necessity or current implementation scope).

## Shapes & semantics

- Matrices are 2D and have `shape == (rows, cols)`.
- Vectors are 1D and have `shape == (n,)`.
- Many objects support transpose as a metadata view (`.T`), which can change the logical shape without copying underlying storage.

## NxM-enabled

### Allocation

- `pycauset.zeros((rows, cols), dtype=...)`
- `pycauset.ones((rows, cols), dtype=...)`
- `pycauset.empty((rows, cols), dtype=...)`

### Dense matrices

Rectangular dense matrices are supported for:

- float16/float32/float64
- int8/int16/int32/int64
- uint8/uint16/uint32/uint64
- complex_float16/complex_float32/complex_float64
- bool/bit (`DenseBitMatrix`, bit-packed)

### Identity-like matrices

- `pycauset.identity(n)` creates an $n \times n$ identity matrix.
- `pycauset.identity([rows, cols])` creates a `rows × cols` identity-like matrix (ones on the diagonal up to $\min(rows, cols)$).
- `pycauset.I(...)` / `pycauset.IdentityMatrix(...)` also support rectangular shapes.

### NumPy interop

- `pycauset.matrix(np_array)` supports rectangular 2D arrays for dense matrices.
- `pycauset.vector(np_array)` supports 1D arrays for vectors.
- `np.asarray(pycauset_matrix)` / `np.array(pycauset_matrix)` returns arrays with matching shape.

### Persistence

- `pycauset.save(obj, path)` and `pycauset.load(path)` preserve `(rows, cols)` and transpose metadata.

### Operations

- Matrix-matrix matmul follows the standard rule:

  $$ (m, k) @ (k, n) \to (m, n) $$

- Elementwise ops (`+`, `-`, `*` elementwise) require exact shape match.

## Restricted / square-only

These operations require square inputs by definition:

- Determinant
- Inverse
- Many eigen/spectral routines

These structures are square-only by definition:

- Triangular matrices (including causal matrices)
- Diagonal matrices
- Symmetric / antisymmetric matrices

## Notes

If you hit a shape-related exception:

- Verify you’re using `rows()/cols()` (or `shape`) rather than relying on legacy “N”.
- For matmul, verify inner dimensions match.

If the status here disagrees with observed behavior, treat it as a bug and update this page along with a regression test.
