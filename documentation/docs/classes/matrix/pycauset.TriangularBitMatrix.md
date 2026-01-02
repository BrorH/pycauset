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
*   **size()** (*int*): Total logical element count ($N \times N$).

## Methods

### Indexing

Element access uses NumPy-style indexing:

*   Read: `x = M[i, j]` (returns `False` if $i \ge j$)
*   Write: `M[i, j] = value` (raises `ValueError` if $i \ge j$)

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

## See also

*   [[docs/functions/pycauset.causal_matrix.md|pycauset.causal_matrix]]
*   [[docs/classes/matrix/pycauset.TriangularIntegerMatrix.md|pycauset.TriangularIntegerMatrix]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
