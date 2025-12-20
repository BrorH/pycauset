# pycauset.dot

```python
pycauset.dot(a: VectorBase, b: VectorBase) -> float | complex
```

Compute the dot product of two vectors.

This is a convenience wrapper around `a.dot(b)`.

Notes:

- For real vectors, the result is a `float`.
- For complex vectors, the result is a `complex`.
- No implicit conjugation is applied. For a Hermitian-style inner product, use `a.conj().dot(b)` or `a.H @ b`.

## Parameters

- `a`: The first vector.
- `b`: The second vector.

## Returns

The dot product of the two vectors.

## Exceptions

- Raises `TypeError` if `a` is not a vector (does not provide `.dot(...)`).
- Raises `ValueError` if vector sizes do not match.

## Examples

```python
import numpy as np
import pycauset as pc

v = pc.vector([1.0, 2.0, 3.0], dtype="float64")
assert pc.dot(v, v) == 14.0

z = pc.ComplexFloat64Vector(np.array([1 + 2j, 3 - 4j], dtype=np.complex128))
assert pc.dot(z, z) == (1 + 2j) * (1 + 2j) + (3 - 4j) * (3 - 4j)
assert pc.dot(z.conj(), z) == np.conj(pc.dot(z, z))
```

## See also

- [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/functions/pycauset.norm.md|pycauset.norm]]
