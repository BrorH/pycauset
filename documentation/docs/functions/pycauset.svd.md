# pycauset.svd

```python
pycauset.svd(a, full_matrices=True, compute_uv=True)
```

Singular value decomposition `a = U @ diag(S) @ Vh`, as NumPy's `linalg.svd`: LAPACK
`?gesdd` in the operand's precision (integers in float64). Any shape.

## Returns

*   With `compute_uv=True`: a tuple `(U, S, Vh)`. `U` is m x m and `Vh` n x n, or m x k and
    k x n with `full_matrices=False` (k = min(m, n)). `S` is a vector of the k singular
    values, descending (real, also for complex input).
*   With `compute_uv=False`: `S` only (also [[docs/functions/pycauset.svdvals.md|pycauset.svdvals]]).

## Example

```python
import pycauset as pc
import numpy as np

a = np.arange(6.0).reshape(2, 3)
U, S, Vh = pc.svd(pc.matrix(a), full_matrices=False)
np.testing.assert_allclose(np.asarray(U) @ np.diag(np.asarray(S)) @ np.asarray(Vh), a, atol=1e-12)
np.testing.assert_allclose(np.asarray(pc.svd(pc.matrix(a), compute_uv=False)), np.linalg.svd(a, compute_uv=False))
```
