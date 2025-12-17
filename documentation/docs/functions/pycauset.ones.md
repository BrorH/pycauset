# pycauset.ones

```python
pycauset.ones(shape, *, dtype, **kwargs)
```

Allocate a vector or matrix filled with ones.

`dtype` is required.

## Parameters

*   **shape** (*int or tuple*):
    *   `n` allocates a length-`n` vector.
    *   `(n,)` allocates a length-`n` vector.
    *   `(n, m)` allocates an `n×m` matrix.

    Notes:
    *   Rectangular allocation is supported for dense numeric matrix types.
    *   `dtype="bool"`/`dtype="bit"` uses bit-packed storage (`DenseBitMatrix`) and supports rectangular `(rows, cols)` shapes.
*   **dtype** (*str or type*): Storage dtype token.
*   **kwargs**: Passed through to the backend allocator.

## Returns

*   **VectorBase or MatrixBase**: A newly allocated object.

## Examples

```python
import pycauset

v = pycauset.ones((5,), dtype="int32")
m = pycauset.ones((3, 7), dtype="float64")
```

## See also

*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.empty.md|pycauset.empty]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[guides/Matrix Guide|Matrix Guide]]
