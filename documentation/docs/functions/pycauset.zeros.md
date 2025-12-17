# pycauset.zeros

```python
pycauset.zeros(shape, *, dtype, **kwargs)
```

Allocate a vector or matrix filled with zeros.

`dtype` is required.

## Parameters

*   **shape** (*int or tuple*):
    *   `n` allocates a length-`n` vector.
    *   `(n,)` allocates a length-`n` vector.
    *   `(n, m)` allocates an `n×m` matrix.

    Notes:
    *   Rectangular allocation is supported for dense numeric matrix types.
    *   `dtype="bool"`/`dtype="bit"` uses bit-packed storage (`DenseBitMatrix`) and supports rectangular `(rows, cols)` shapes.
*   **dtype** (*str or type*): Storage dtype token (e.g. `"float64"`, `"int32"`, `float`, `int`).
*   **kwargs**: Passed through to the backend allocator.

## Returns

*   **VectorBase or MatrixBase**: A newly allocated object.

## Examples

```python
import pycauset

v = pycauset.zeros(10, dtype="float64")
m = pycauset.zeros((128, 64), dtype="float32")
```

## See also

*   [[docs/functions/pycauset.ones.md|pycauset.ones]]
*   [[docs/functions/pycauset.empty.md|pycauset.empty]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[guides/Matrix Guide|Matrix Guide]]
