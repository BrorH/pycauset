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

Current implementation is an endpoint-first baseline that uses `invert(a) @ b` when no dedicated solver is available.

## Examples

```python
import pycauset as pc

A = pc.matrix(((4.0, 1.0), (2.0, 3.0)))
b = pc.vector((1.0, 0.0))

x = pc.solve(A, b)
```
