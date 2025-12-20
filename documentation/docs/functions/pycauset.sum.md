# pycauset.sum

```python
pycauset.sum(x) -> complex
```

Return the sum of all elements in a vector or matrix.

## Parameters

- `x`: A vector or matrix.

## Returns

- `complex`: The total sum.

Notes:

- For real inputs, the result is returned as `complex` with zero imaginary part.
- For complex inputs, the full complex sum is returned.
- Conjugated views (`x.conj()` / `x.H`) are respected.

## Exceptions

- Raises `TypeError` if `x` is not a `VectorBase` or `MatrixBase`.

## Examples

```python
import numpy as np
import pycauset as pc

v = pc.FloatVector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
assert pc.sum(v) == 6.0 + 0.0j

z = pc.ComplexFloat64Vector(np.array([1 + 2j, 3 - 4j], dtype=np.complex128))
assert pc.sum(z) == (1 + 2j) + (3 - 4j)
assert pc.sum(z.conj()) == np.conj(pc.sum(z))
```

## See also

- [[docs/functions/pycauset.norm.md|pycauset.norm]]
- [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
- [[docs/functions/pycauset.vector.md|pycauset.vector]]
