# pycauset.TriangularFloatMatrix

```python
class pycauset.TriangularFloatMatrix(n)
```

A memory-mapped matrix containing 64-bit double-precision floating point values. It is strictly upper triangular. This class is returned by [[pycauset.compute_k]].

Inherits from [[pycauset.TriangularMatrix]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `get(i, j)`
Returns the value at row `i` and column `j`.
*   Returns `0.0` if $i \ge j$.

### `set(i, j, value)`
Sets the value at row `i` and column `j`.
*   Raises `ValueError` if $i \ge j$ (strictly upper triangular).

### `close()`
Closes the memory-mapped file handle.

### `get_element_as_double(i, j)`
Returns the element as a double, applying the internal scalar.
*   Optimized to avoid multiplication if `scalar == 1.0`.

### `invert()`
Returns a new `TriangularFloatMatrix` with bitwise-inverted elements.

## Properties
