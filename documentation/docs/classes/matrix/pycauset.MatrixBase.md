# pycauset.MatrixBase

```python
class pycauset.MatrixBase
```

The abstract base class for all matrix types in `pycauset`. It manages the storage (memory-mapped file or RAM), lifecycle, and common properties like scaling.

## Properties

*   **scalar** (*float*): A scaling factor applied to all elements when accessed as doubles. Defaults to `1.0`. Setting this property updates the file header instantly.
*   **seed** (*int*): The random seed used to generate the matrix, if applicable. Read-only. Returns `0` if no seed was recorded.
*   **is_temporary** (*bool*): Indicates whether the backing storage is temporary (RAM or temp file) and should be deleted/released on exit.
*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.

## Methods

### `close()`
Releases the memory-mapped file handle or frees the RAM buffer. The matrix object becomes unusable after calling this method.

### `get_backing_file()`
Returns the absolute path to the backing file on disk.

### `get_element_as_double(i, j)`
Returns the element at $(i, j)$ as a double-precision float, multiplied by `scalar`.

### `trace()`
Returns the trace of the matrix (sum of diagonal elements).
*   **Caching**: The result is cached in memory. Subsequent calls return the cached value instantly.
*   **Persistence**: When the matrix is saved using `save()`, the cached trace is written to `metadata.json` and automatically restored upon loading.

### `determinant()`
Returns the determinant of the matrix.
*   **Caching**: The result is cached in memory.
*   **Persistence**: Automatically saved to `metadata.json` and restored upon loading.

### `eigenvalues()`
Returns the eigenvalues of the matrix as a list of complex numbers.
*   **Returns**: A list of `complex` (Python) or a `ComplexVector` (internal).
*   **Caching**: The result is cached in memory.
*   **Persistence**: Automatically saved to `metadata.json` and restored upon loading.

### `eigenvectors(save=False)`
Computes the eigenvectors of the matrix.
*   **save** (*bool*): If `True`, the computed eigenvectors are **appended** to the matrix's backing `.pycauset` ZIP file as binary files (`eigenvectors.real.bin`, `eigenvectors.imag.bin`). This allows them to be loaded instantly in future sessions without recomputation. Defaults to `False`.
*   **Returns**: A `ComplexMatrix` where columns are the eigenvectors.
*   **Caching**: The result is cached in memory for the lifetime of the object.

### `inverse(save=False)`
Computes the inverse of the matrix.
*   **save** (*bool*): If `True`, the computed inverse matrix is **appended** to the matrix's backing `.pycauset` ZIP file as `inverse.bin`. Defaults to `False`.
*   **Returns**: A `FloatMatrix` (Dense) representing the inverse.
*   **Caching**: The result is cached in memory for the lifetime of the object.

### `save(path)`
Saves the matrix to a permanent location.
*   **path** (*str*): The destination path.
*   This creates a hard link if possible, or copies the file.
