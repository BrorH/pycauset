# pycauset.ones

```python
pycauset.ones(shape, *, dtype=None, **kwargs)
```

Allocate a vector or matrix filled with ones.

When `dtype` is omitted, the result is a dtype-deferred wrapper (all-equal
"constant" structure, value `1`) that resolves to `int32` on first use, or to
the dtype of the first value written into it. Passing `dtype` allocates a
concrete object immediately.

## Parameters

*   **shape** (*int or tuple*):
    *   `n` allocates a length-`n` vector.
    *   `(n,)` allocates a length-`n` vector.
    *   `(n, m)` allocates an `n×m` matrix.

    Notes:
    *   Rectangular allocation is supported for dense numeric matrix types.
    *   `dtype="bool"`/`dtype="bit"` uses bit-packed storage (`DenseBitMatrix`) and supports rectangular `(rows, cols)` shapes.
*   **dtype** (*str or type, optional*): Storage dtype token. Defaults to a deferred `int32` when omitted.
*   **kwargs**: Passed through to the backend allocator.

## Returns

*   **VectorBase or MatrixBase**: A newly allocated object. With no `dtype`, a dtype-deferred wrapper is returned instead and materializes on first use.

## Examples

```python
import pycauset

v = pycauset.ones(5)                   # dtype-deferred; resolves to int32
m = pycauset.ones((3, 7), dtype="float64")
```

## See also

*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.empty.md|pycauset.empty]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[guides/Matrix Guide|Matrix Guide]]
