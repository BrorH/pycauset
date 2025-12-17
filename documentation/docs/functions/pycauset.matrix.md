# pycauset.matrix

```python
pycauset.matrix(source, dtype=None, **kwargs)
```

Create a 2D matrix from matrix-like input.

This is a data constructor (aligned with `np.array(...)` semantics). 

If `source` is 2D, returns a matrix. If `source` is 1D, returns a vector.

Rectangular shapes are supported for dense matrices, including numeric dtypes (int/uint/float/complex) and boolean/bit matrices. Boolean 2D inputs use bit-packed storage (`DenseBitMatrix`).

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D nested data (e.g. list) / 1D NumPy array, or 2D nested data (e.g. list-of-lists) / 2D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`).
*   **kwargs**: Passed through to the backend constructor.

## Returns

*   **MatrixBase or VectorBase**: A concrete matrix (for 2D input) or vector (for 1D input).

## Examples

```python
import pycauset

m = pycauset.matrix(((1, 2), (3, 4)))

# 1D input returns a vector
v = pycauset.matrix((1, 2, 3))

# Coerce dtype
m_f32 = pycauset.matrix(((1, 2), (3, 4)), dtype="float32")
```
