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

Property-aware shortcuts (R1_PROPERTIES):
- If `a.properties["is_identity"]` is asserted, the solver returns `b` directly (square only).
- If `a.properties["is_zero"]` is asserted, the solver raises a singularity error.
- If `is_diagonal` / `is_upper_triangular` / `is_lower_triangular` is asserted, the solver routes to `solve_triangular`.

Otherwise, the endpoint-first baseline uses `invert(a) @ b` when no dedicated solver is available.

## Examples

```python
import pycauset as pc

A = pc.matrix(((4.0, 1.0), (2.0, 3.0)))
b = pc.vector((1.0, 0.0))

x = pc.solve(A, b)
```
