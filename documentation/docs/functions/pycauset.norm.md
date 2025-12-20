# pycauset.norm

```python
pycauset.norm(x) -> float
```

Compute the norm of a vector or matrix.

- **Vector input**: returns the $\ell_2$ (Euclidean) norm.
- **Matrix input**: returns the **Frobenius** norm.

## Parameters

- `x`: A vector or matrix.

## Returns

- `float`: The computed norm.

## Exceptions

- Raises `TypeError` if `x` is not a `VectorBase` or `MatrixBase`.

## Examples

```python
import pycauset as pc

v = pc.vector([3.0, 4.0], dtype="float64")
assert pc.norm(v) == 5.0

A = pc.matrix(
    [
        [3.0, 4.0],
        [0.0, 0.0],
    ],
    dtype="float64",
)
assert pc.norm(A) == 5.0
```

## See also

- [[docs/functions/pycauset.dot.md|pycauset.dot]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/functions/pycauset.sum.md|pycauset.sum]]
