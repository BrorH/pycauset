# Release 1: what shipped

Release 1 is the foundation slice of PyCauset: rectangular matrices, a real dtype
system (including complex floats), a persistent on-disk container, and a semantic
`properties` mechanism that can change which algorithm runs.

Everything below is the implemented, user-facing behavior. The planning docs are
archived under `internals/plans/archive/`.

## Shapes: vectors and matrices

- Vectors are 1D, matrices are 2D.
- Rectangular dense matrices work end-to-end (allocation, indexing, transpose
  views, NumPy interop, persistence).

Constructors take **data**; allocators take **shapes**:

```python
import pycauset as pc

m = pc.matrix(((1, 2), (3, 4)))   # data -> matrix
v = pc.matrix((1, 2, 3))          # data -> vector
z = pc.zeros((2, 3), dtype="float64")   # shape -> allocated
```

`matrix(...)` does not treat a tuple as a shape. Use `zeros`/`ones`/`empty` for
size-first allocation (`dtype` is required there).

NumPy alignment: `shape == (rows, cols)` for matrices, `(n,)` for vectors.
`size()` is the total element count, `len(x)` is the first dimension.

Transpose is usually a zero-copy metadata view (`A.T`), not a copy.

Two things stay square-only:

- Structures that are square by definition (triangular, diagonal,
  symmetric/antisymmetric).
- Operations that are square by math (determinant, inverse, most spectral
  routines).

See [[guides/NxM Support.md|NxM Support Status]] for the current list.

## Storage: the `.pycauset` container

Release 1 ships a single-file `.pycauset` container that is mmap-friendly,
crash-consistent, and forward-compatible.

```python
import pycauset as pc

A = pc.zeros((128, 64), dtype="float32")
pc.save(A, "A.pycauset")
B = pc.load("A.pycauset")
```

A `.pycauset` file is an **immutable snapshot**:

- `load(path)` gives you a snapshot-backed object.
- Mutating it does not write back to `path`.
- To persist changes, save a new snapshot.

Persistence round-trips identity metadata (shape, dtype, type, layout), view-state
metadata (transpose, conjugation, scalar), semantic properties, and cached-derived
values. Transpose is preserved as metadata, not densified.

Block matrices persist as a manifest plus a sidecar directory of child files
(`path + ".blocks"`). They save blockwise (never a global densify), and mixed
snapshots fail deterministically via `payload_uuid` pins.

The on-disk format spec is [[dev/PyCauset Container Format.md|here]]; the
user-facing semantics are in [[guides/Storage and Memory.md|Storage and Memory]].

## Properties: semantic metadata

`obj.properties` is a typed mapping of **semantic assertions** and
**cached-derived values**.

!!! warning "Assertions are gospel"
    If you mark a matrix diagonal/triangular/unitary, PyCauset is allowed to run
    algorithms that assume it is true. It does not scan the payload to verify you.

Keys use tri-state behavior: **unset** (absent), **True** (asserted), **False**
(explicitly negated). Explicit `False` is different from unset.

Common assertion keys: `is_zero`, `is_identity`, `is_diagonal`,
`is_upper_triangular`, `is_lower_triangular`, `is_symmetric`, `is_hermitian`,
`is_unitary`, and more. Cached-derived keys include `trace`, `determinant`,
`norm`, `sum`, `eigenvalues` (validity-checked and invalidated on mutation).

```python
A = pc.identity(3)
A.properties["is_upper_triangular"] = True
b = pc.vector((1.0, 2.0, 3.0))
x = pc.solve_triangular(A, b)   # solver trusts the assertion
```

Unset by deleting the key or assigning `None`.

Property propagation under metadata-only transforms is deterministic: transpose
swaps upper/lower triangular, conjugation conjugates `diagonal_value` and cached
complex values, scalar scale updates cached values where there is a safe O(1)
rule.

A few endpoints are property-aware: `solve` short-circuits `is_identity` and
rejects `is_zero`; `matmul` may exploit diagonal/triangular claims; `eigvalsh`
consults cached eigenvalues and rejects explicit `is_hermitian=False`.

## DTypes, promotion, overflow

The supported scalar set: `bool`/`bit`, `int8`-`int64`, `uint8`-`uint64`,
`float16`/`float32`/`float64`, `complex_float16`/`complex_float32`/`complex_float64`.
Complex is float-only (no complex int or complex bit).

Mixed float precision **underpromotes** by default: `float32` with `float64` uses
`float32` compute and storage, and warns.

Integer overflow:

- Elementwise (`add`, `sub`, `mul`, `div`): wraps silently, like C/NumPy.
- Reductions (`matmul`): uses a wider internal accumulator and raises
  `OverflowError` if the result does not fit the output dtype. `dot` returns a
  Python `float`.

Large integer matmuls may emit a conservative `PyCausetOverflowRiskWarning`.

`bit` is special: each operation must say what it means on `bit` (bitwise, numeric
widen, or error by design), because widening a huge bit dataset can be infeasible.

## Linear algebra

The R1 surface, all behind a single routing boundary:

- Matmul/dot: `matmul`, `dot`
- Solves: `solve`, `solve_triangular`, `lstsq`
- Factorizations: `lu`, `cholesky`
- Spectral/SVD: `eig`, `eigh`, `eigvalsh`, `svd`, `pinv`, `cond`, `slogdet`

Dense matmul follows `(m, k) @ (k, n) -> (m, n)`. Inverse/determinant and most
eigen routines are square-only.

Block matrices: `matrix(...)` builds one when given a 2D grid of matrices. Once
block, always block (`@`, `+`, `-`, `*`, `/` return block results). Outputs are
thunked per block; they evaluate on element access, dense conversion, persistence,
or crossing the compute boundary.

## See also

- [[guides/NxM Support.md|NxM Support Status]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[internals/DType System.md|DType System]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[docs/index.md|API Reference]]
