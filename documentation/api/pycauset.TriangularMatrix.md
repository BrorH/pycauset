# pycauset.TriangularMatrix

```python
class pycauset.TriangularMatrix(n)
```

Base class for all strictly upper triangular matrices. It manages the row-offset calculations required to map 2D coordinates $(i, j)$ where $i < j$ into a linear memory space.

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Methods

### `get_row_offset(i)`
Returns the byte offset for the start of row `i` in the backing storage.

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.
