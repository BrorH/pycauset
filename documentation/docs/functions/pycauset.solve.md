# pycauset.solve

```python
pycauset.solve(a, b)
```

Solve the linear system:

$$
A X = B
$$

## Parameters

* **a** (*MatrixBase*): Coefficient matrix $A$.
* **b** (*VectorBase or MatrixBase*): Right-hand side $B$.

## Returns

* **VectorBase or MatrixBase**: The solution $X$.

## Notes

NumPy's algorithm, `np.linalg.solve`: LU with partial pivoting (LAPACK `?getrf` and triangular
solves) in the operands' common LAPACK dtype (integers in float64). Any structure, view and
dtype; structured operands are written out first. `b` may be a matrix or a vector (the
result has `b`'s kind). A singular `a` raises `ValueError` (NumPy: `LinAlgError`, a `ValueError`).

## Examples

```python
import pycauset as pc

A = pc.matrix(((4.0, 1.0), (2.0, 3.0)))
b = pc.vector((1.0, 0.0))

x = pc.solve(A, b)
```
