# pycauset.sum

```python
pycauset.sum(x) -> int | float | complex
```

Return the sum of all elements in a vector or matrix, like `numpy.sum`.

## Parameters

- `x`: A vector or matrix.

## Returns

- `int` for integer and bool inputs: exact (no wrap-around; `OverflowError` only beyond
  128 bits). For a causal matrix `C` this is the number of relations, counted on the bits.
- `float` for real float inputs, `complex` for complex inputs.

Notes:

- Views are respected: scaled views (`3 * A`, integer elements wrap first, as in NumPy),
  transposes, slices, conjugated views (`x.conj()` / `x.H`), and lazy closures
  (`pc.sum(C - C.T)` is computed from `C`'s bit count without writing anything out).
- Float sums are deterministic: same result for any thread count (fixed summation order).

## Exceptions

- Raises `TypeError` if `x` is not a `VectorBase` or `MatrixBase`.

## Examples

```python
import numpy as np
import pycauset as pc

v = pc.vector(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype="float64")
assert pc.sum(v) == 6.0
assert pc.sum(pc.vector([1, 2, 3])) == 6  # an int

z = pc.vector(np.array([1 + 2j, 3 - 4j], dtype=np.complex128), dtype="complex_float64")
assert pc.sum(z) == (1 + 2j) + (3 - 4j)
assert pc.sum(z.conj()) == np.conj(pc.sum(z))
```

## See also

- [[docs/functions/pycauset.norm.md|pycauset.norm]]
- [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
- [[docs/functions/pycauset.vector.md|pycauset.vector]]
