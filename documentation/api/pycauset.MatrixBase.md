# pycauset.MatrixBase

```python
class pycauset.MatrixBase
```

The abstract base class for all matrix types in `pycauset`. It manages the memory-mapped backing file, lifecycle, and common properties like scaling.

## Properties

*   **scalar** (*float*): A scaling factor applied to all elements when accessed as doubles. Defaults to `1.0`. Setting this property updates the file header instantly.
*   **seed** (*int*): The random seed used to generate the matrix, if applicable. Read-only. Returns `0` if no seed was recorded.
*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `close()`
Releases the memory-mapped file handle. The matrix object becomes unusable after calling this method.

### `get_backing_file()`
Returns the absolute path to the backing file on disk.

### `get_element_as_double(i, j)`
Returns the element at $(i, j)$ as a double-precision float, multiplied by `scalar`.
