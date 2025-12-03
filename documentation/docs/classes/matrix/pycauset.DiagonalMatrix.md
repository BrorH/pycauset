# pycauset.DiagonalMatrix

```python
class pycauset.DiagonalMatrix(n, backing_file="")
```

A memory-efficient representation of a Diagonal Matrix. It stores $N$ elements linearly in memory (or on disk).

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).
*   **backing_file** (*str*, optional): Path to a file for memory mapping. If not provided, a temporary file is used.

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `get(i, j)`
Returns the value at row `i` and column `j`.
*   Returns the diagonal element $D_i$ if $i == j$.
*   Returns `0.0` (or equivalent zero) if $i \neq j$.

### `set(i, j, value)`
Sets the value at row `i` and column `j`.
*   If $i == j$, updates the diagonal element $D_i$.
*   If $i \neq j$, throws an error if `value` is not zero.

### `get_diagonal(i)`
Returns the diagonal element at index `i`.

### `set_diagonal(i, value)`
Sets the diagonal element at index `i`.

### `multiply(other)`
Multiplies this matrix by another matrix.
*   Optimized multiplication when `other` is also a `DiagonalMatrix` or `IdentityMatrix`.

### `multiply_scalar(factor)`
Multiplies the matrix by a scalar factor.
*   **factor** (*float*): The value to multiply by.
*   **Returns**: A new `DiagonalMatrix` (or subclass) with updated scalar.
