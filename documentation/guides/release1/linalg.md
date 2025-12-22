# R1 Linear Algebra (Core Ops)

Release 1 ships a “foundation” linear algebra surface with stable Python endpoints and a single routing boundary (so future CPU/GPU/out-of-core optimizations can land behind the same entry points).

This guide is about *what you can call* and *what guarantees exist*, not which backend is currently fastest.

## Minimal example

```python
import pycauset as pc

A = pc.matrix(((4.0, 1.0), (2.0, 3.0)))
b = pc.vector((1.0, 0.0))

x = pc.solve(A, b)
```

## Core endpoints in Release 1

The Release 1 linalg surface includes these function families:

- Matmul / dot:
  - [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
  - [[docs/functions/pycauset.dot.md|pycauset.dot]]

- Solves:
  - [[docs/functions/pycauset.solve.md|pycauset.solve]]
  - [[docs/functions/pycauset.solve_triangular.md|pycauset.solve_triangular]]
  - [[docs/functions/pycauset.lstsq.md|pycauset.lstsq]]

- Factorizations:
  - [[docs/functions/pycauset.lu.md|pycauset.lu]]
  - [[docs/functions/pycauset.cholesky.md|pycauset.cholesky]]

- Spectral / SVD / conditioning:
  - [[docs/functions/pycauset.eig.md|pycauset.eig]]
  - [[docs/functions/pycauset.eigh.md|pycauset.eigh]]
  - [[docs/functions/pycauset.eigvalsh.md|pycauset.eigvalsh]]
  - [[docs/functions/pycauset.svd.md|pycauset.svd]]
  - [[docs/functions/pycauset.pinv.md|pycauset.pinv]]
  - [[docs/functions/pycauset.cond.md|pycauset.cond]]
  - [[docs/functions/pycauset.slogdet.md|pycauset.slogdet]]

## Shape constraints

- Dense matmul follows NxM rules: `(m, k) @ (k, n) -> (m, n)`.
- Many routines are **square-only** by definition (inverse/determinant and most eigen routines).

See [[guides/NxM Support.md|NxM Support Status]] and [[dev/Square-only Assumptions.md|Square-only Assumptions]].

## Property-aware behavior (R1_PROPERTIES)

Some linalg endpoints consult `A.properties`:

- `solve`:
  - returns `b` when `is_identity=True` (square only)
  - rejects `is_zero=True`
  - routes diagonal/triangular claims to `solve_triangular`

- `solve_triangular` treats triangular/diagonal claims as gospel and does not truth-validate.

See [[guides/release1/properties.md|R1 Properties]] for the rules and failure modes.

## Practical example (property-driven solve)

```python
import pycauset as pc

A = pc.identity(3)
A.properties["is_upper_triangular"] = True
b = pc.vector((1.0, 2.0, 3.0))

x = pc.solve(A, b)  # routes to solve_triangular under the hood
```

## See also

- [[guides/Linear Algebra Operations.md|Linear Algebra Operations]]
- [[guides/release1/properties.md|R1 Properties]]
- [[docs/functions/pycauset.solve.md|pycauset.solve]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[internals/Compute Architecture.md|Compute Architecture]]
