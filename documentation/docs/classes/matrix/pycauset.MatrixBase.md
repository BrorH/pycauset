# pycauset.MatrixBase

```python
class pycauset.MatrixBase
```

The abstract base class for all matrix types in `pycauset`. It manages the storage (memory-mapped file or RAM), lifecycle, and common properties like scaling.

`MatrixBase` is **rectangular-aware**: every matrix has a logical `(rows, cols)` shape, and transpose is usually represented as a metadata view.

## Shape and size

*   `shape` is a property returning `(rows, cols)`.
*   `rows()` / `cols()` are methods returning the logical dimensions.
*   `size()` is a method returning the total element count: $\text{rows} \times \text{cols}$.

## Elementwise operations and broadcasting

Elementwise operators (`+`, `-`, `*`, `/`) follow **NumPy-style 2D broadcasting**:

*   Two shapes `(a_rows, a_cols)` and `(b_rows, b_cols)` are compatible if each dimension is either equal or one of them is `1`.
*   The result shape is `(max(a_rows, b_rows), max(a_cols, b_cols))`.
*   When mixing a matrix with a **1D NumPy array** in an elementwise operation, the array is treated as a **row vector** with shape `(1, n)`.

## Properties

*   **scalar** (*float* or *complex*): A scaling factor applied to all elements when accessed as doubles. Defaults to `1.0`. Setting this property updates the file header instantly.
*   **seed** (*int*): The random seed used to generate the matrix, if applicable. Read-only. Returns `0` if no seed was recorded.
*   **is_temporary** (*bool*): Indicates whether the backing storage is temporary (RAM or temp file) and should be deleted/released on exit.
*   **shape** (*tuple[int, int]*): The dimensions of the matrix `(rows, cols)`.
*   **backing_file** (*str*): Absolute path to the backing file on disk.

## Methods

### `rows()` / `cols()` / `size()`

*   `rows()` and `cols()` report the logical shape.
*   `size()` reports the total element count.

### `close()`
Releases the memory-mapped file handle or frees the RAM buffer. The matrix object becomes unusable after calling this method.

### `get_backing_file()`
Returns the absolute path to the backing file on disk.

### `get_element_as_double(i, j)`
Returns the element at $(i, j)$ as a double-precision float, multiplied by `scalar`.

### `trace()`
Returns the trace of the matrix (sum of diagonal elements).

For rectangular matrices, this uses the diagonal length $\min(\text{rows}, \text{cols})$.
*   **Caching**: The result is cached in memory. Subsequent calls return the cached value instantly.
*   **Persistence**: When the matrix is saved using `save()`, the cached trace is written to `metadata.json` and automatically restored upon loading.

### `determinant()`
Returns the determinant of the matrix.

This is a **square-only** operation and will raise for non-square shapes.
*   **Caching**: The result is cached in memory.
*   **Persistence**: Automatically saved to `metadata.json` and restored upon loading.

### `transpose()` / `T`
Returns a transposed view of the same underlying storage (usually metadata-only; no element-wise copy).

### `get(i, j)` / `set(i, j, value)`
Element access. Equivalent to `m[i, j]` and `m[i, j] = value`.

## Examples

```python
import pycauset as pc

A = pc.zeros((2, 3), dtype="float64")
assert A.shape == (2, 3)
assert A.rows() == 2
assert A.cols() == 3
assert A.size() == 6

AT = A.T
assert AT.shape == (3, 2)
```

## See also

*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.ones.md|pycauset.ones]]
*   [[docs/functions/pycauset.empty.md|pycauset.empty]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[internals/Memory and Data|Memory and Data]]
*   [[guides/Matrix Guide|Matrix Guide]]
