# pycauset.IntegerMatrix

```python
class pycauset.IntegerMatrix(n)
```

A memory-mapped matrix containing 32-bit integer values. It is a dense matrix (stores all $N \times N$ elements). This class is typically returned by matrix multiplication operations like [[pycauset.matmul]].

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `get(i, j)`
Returns the integer value at row `i` and column `j`.

### `set(i, j, value)`
Sets the integer value at row `i` and column `j`.

### `get_element_as_double(i, j)`
Returns the element as a double, applying the internal scalar. Optimized for `scalar == 1.0`.
