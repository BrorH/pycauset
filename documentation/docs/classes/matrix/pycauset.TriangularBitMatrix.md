# pycauset.TriangularBitMatrix

```python
class pycauset.TriangularBitMatrix(n)
```

The primary class for representing causal structures. It stores boolean values in a strictly upper triangular format ($i < j$), using bit-packing for efficiency (1 bit per element).

Inherits from [[pycauset.TriangularMatrix]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `get(i, j)`
Returns the boolean value at row `i` and column `j`.
*   Returns `False` if $i \ge j$.

### `set(i, j, value)`
Sets the value at row `i` and column `j`.
*   Raises `ValueError` if $i \ge j$ (strictly upper triangular).

### `multiply(other)`
Multiplies this matrix by another `TriangularBitMatrix`.
*   **other**: Another `TriangularBitMatrix`.
*   **Returns**: A [[pycauset.TriangularIntegerMatrix]].
*   **Performance Note**: This operation is highly optimized on the CPU using `popcount` instructions. It is typically faster than GPU execution for sparse boolean matrices.

### `elementwise_multiply(other)`
Performs elementwise logical AND.
*   **Returns**: A new `TriangularBitMatrix`.

### `get_element_as_double(i, j)`
Returns the element as a double, applying the internal scalar.
*   Optimized to avoid multiplication if `scalar == 1.0`.
