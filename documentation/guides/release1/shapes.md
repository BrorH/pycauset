# R1 Shapes (NxM Matrices)

This guide explains the Release 1 shape model: **vectors are 1D**, **matrices are 2D**, and **rectangular (rows×cols) dense matrices are supported end-to-end** (allocation, indexing, transpose views, NumPy interop, and persistence).

## Goal

- Create vectors and matrices with NumPy-like shape semantics.
- Use rectangular dense matrices safely.
- Know what is still square-only.

## Minimal example

```python
import pycauset as pc

A = pc.zeros((2, 3), dtype="float64")
B = pc.ones((3, 4), dtype="float64")
C = A @ B

assert A.shape == (2, 3)
assert B.shape == (3, 4)
assert C.shape == (2, 4)
```

## Constructors vs allocators

### Data constructors (`matrix`, `vector`)

- [[docs/functions/pycauset.matrix.md|pycauset.matrix]] constructs from **data** (aligned with `np.array`).
- [[docs/functions/pycauset.vector.md|pycauset.vector]] constructs from **data**.

Important: these constructors do **not** interpret tuples as shapes.

```python
import pycauset as pc

m = pc.matrix(((1, 2), (3, 4)))   # 2D data -> matrix
v = pc.matrix((1, 2, 3))         # 1D data -> vector
```

### Shape allocators (`zeros`, `ones`, `empty`)

Use shape-based allocation when you want size-first creation:

- [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
- [[docs/functions/pycauset.ones.md|pycauset.ones]]
- [[docs/functions/pycauset.empty.md|pycauset.empty]]

`dtype` is required for these APIs.

```python
import pycauset as pc

v = pc.zeros((10,), dtype="float32")
M = pc.empty((128, 64), dtype="int16")
```

`empty(...)` does not guarantee initialization; see the API page for details.

## Shape, size, and length

Release 1 aligns these basics with NumPy:

- Matrices: `shape == (rows, cols)`
- Vectors: `shape == (n,)`
- `size()` returns the **total element count**.
- `len(x)` is the first dimension: `rows` for matrices, `n` for vectors.

See [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]] and [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]].

## Transpose is usually a metadata view

Most dense objects support transpose as a **zero-copy view**:

```python
import pycauset as pc

A = pc.zeros((2, 3), dtype="float64")
AT = A.T
assert AT.shape == (3, 2)
```

## What is still square-only

Two distinct categories remain square-only:

- **Square-only structures** (by definition): triangular/causal, diagonal, symmetric/antisymmetric.
- **Square-only operations** (by math/implementation): determinant, inverse, many spectral routines.

For the current status list, see:

- [[guides/NxM Support.md|NxM Support Status]]
- [[dev/Square-only Assumptions.md|Square-only Assumptions]]

## See also

- [[guides/NxM Support.md|NxM Support Status]]
- [[guides/Matrix Guide.md|Matrix Guide]]
- [[guides/Vector Guide.md|Vector Guide]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
- [[docs/functions/pycauset.vector.md|pycauset.vector]]
