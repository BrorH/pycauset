# pycauset.FloatMatrix

```python
class pycauset.FloatMatrix(n)
```

A dense $N \times N$ matrix storing 64-bit floating-point values (`double`). Unlike triangular matrices, this class stores the full square matrix.

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Methods

### `get(i, j)`
Returns the value at row `i` and column `j`.

### `set(i, j, value)`
Sets the value at row `i` and column `j`.

### `get_element_as_double(i, j)`
Returns the element as a double, applying the internal scalar.
*   Optimized to avoid multiplication if `scalar == 1.0`.

### `invert()`
Returns a new `FloatMatrix` with bitwise-inverted elements.

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.
