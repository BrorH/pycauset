# pycauset.lstsq

```python
pycauset.lstsq(a, b)
```

Compute a least-squares solution $x$ that approximately minimizes:

$$
\|Ax - b\|_2
$$

## Parameters

* **a** (*MatrixBase*): Coefficient matrix.
* **b** (*VectorBase or MatrixBase*): Right-hand side.

## Returns

* **VectorBase or MatrixBase**: The solution $x$.

## Notes

This is an endpoint-first baseline.

- It currently returns only `x` (unlike `numpy.linalg.lstsq`, which returns a tuple).
- The baseline implementation uses normal equations: $x = (A^T A)^{-1} A^T b$.
