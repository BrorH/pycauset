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

- Returns only `x`, NumPy's `lstsq(a, b)[0]` (NumPy returns a tuple with the residuals, rank
  and singular values too).
- NumPy's algorithm: LAPACK `?gelsd` with NumPy's default cutoff (singular values at or below
  `eps * max(m, n) * max(s)` count as zero), so a rank-deficient `a` gives the minimum-norm
  solution. Integers compute in float64, single precision stays single. Any structure and
  view.

## Example

```python
import pycauset as pc
import numpy as np

a = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, 2.0, 2.0])
x = pc.lstsq(pc.matrix(a), pc.vector(b))
np.testing.assert_allclose(np.asarray(x), np.linalg.lstsq(a, b, rcond=None)[0])
```
