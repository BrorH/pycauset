# pycauset.solve_triangular

```python
pycauset.solve_triangular(a, b)
```

Solves $A X = B$ for a triangular, diagonal or identity $A$: a
[[docs/classes/matrix/pycauset.TriangularMatrix.md|TriangularMatrix]] (or a view of one, such
as its transpose), a `DiagonalMatrix` or an `IdentityMatrix`. Any other matrix raises
`ValueError`; use [[docs/functions/pycauset.solve.md|pycauset.solve]] for general systems.

The answer is the same as `pycauset.solve(a, b)` (LU in NumPy's LAPACK dtype; a singular
matrix, e.g. a strictly triangular one, raises).

## Examples

```python
import pycauset as pc

A = pc.TriangularMatrix(3, dtype="float64", has_diagonal=True)
A[0, 0], A[1, 1], A[2, 2], A[0, 2] = 1.0, 2.0, 4.0, 1.0
x = pc.solve_triangular(A, pc.vector([1.0, 2.0, 3.0]))
```
