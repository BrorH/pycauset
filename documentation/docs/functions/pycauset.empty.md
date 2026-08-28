# pycauset.empty

```python
pycauset.empty(shape, *, dtype=None, **kwargs)
```

Allocate a vector or matrix without guaranteeing initialization.

Note: for some backends this may still be zero-initialized.

When `dtype` is omitted, the result is a dtype-deferred wrapper with no dtype
at all. Its dtype is deduced from the first value written into it (`set`,
`fill`, or item assignment). Reading from, or operating on, a still-typeless
`empty` raises `TypeError` (no silent wrong answer).

## Parameters

*   **shape** (*int or tuple*):
    *   `n` allocates a length-`n` vector.
    *   `(n,)` allocates a length-`n` vector.
    *   `(n, m)` allocates an `n×m` matrix.

    Notes:
    *   Rectangular allocation is supported for dense numeric matrix types.
    *   `dtype="bool"`/`dtype="bit"` uses bit-packed storage (`DenseBitMatrix`) and supports rectangular `(rows, cols)` shapes.
*   **dtype** (*str or type, optional*): Storage dtype token. When omitted, the dtype is deduced from the first written value.
*   **kwargs**: Passed through to the backend allocator.

## Returns

*   **VectorBase or MatrixBase**: A newly allocated object. With no `dtype`, a dtype-deferred wrapper is returned instead and materializes on first write.

## Examples

```python
import pycauset as pc

tmp = pc.empty((256, 64), dtype="float32")

late = pc.empty((4, 4))   # dtype-deferred
late.fill(2.5)            # resolves to float64
```

## See also

*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.ones.md|pycauset.ones]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[guides/Matrix Guide|Matrix Guide]]
