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

## Indexing, slicing, and assignment

`MatrixBase` implements NumPy-aligned 2D indexing for dense matrices (basic indexing as views, advanced indexing as copies). Structured/triangular matrices currently reject slicing.

### Reads (`__getitem__`)

* **Basic indexing (views):** integers (with negative wrap), `:`, `slice` with step (positive/negative), and `...` return a view that shares backing storage when both row/col steps are `1`. Transpose/conjugate metadata is preserved.
* **Advanced indexing (copies):** 1D integer arrays (negative wrap) and 1D boolean masks per axis are supported. Any use of arrays (alone or mixed with basic) returns a copy with NumPy shape rules. Two array axes must have equal length or length-1 to broadcast; otherwise an error is raised.
* **Empty and OOB:** Empty slices are allowed; out-of-bounds indices raise `IndexError`.
* **Not supported:** `None`/newaxis is rejected (matrices stay 2D-only).

### Writes (`__setitem__`)

* **Right-hand side forms:** scalar, NumPy arrays (0D/1D/2D), or another dense matrix.
* **Broadcasting:** RHS must broadcast to the indexed region using NumPy 2D rules; otherwise a `ValueError` is raised.
* **Dtype/overflow warnings:** Casting RHS arrays to the target dtype raises `PyCausetDTypeWarning`; narrowing or float→int casts also raise `PyCausetOverflowRiskWarning`.
* **View vs copy:** Basic targets write through views (shared backing). Advanced targets write into the selected elements of the original matrix (index arrays are copies of indices, not of data).
* **Not supported:** `None`/newaxis targets; structured/triangular matrix slicing.

### Interaction with compute kernels

Views that include storage offsets are rejected by `matmul`, `qr`, `lu`, and `inverse` until offset-aware kernels land. Materialize with `copy()` before calling those ops.

## Properties

*   **scalar** (*float* or *complex*): A scaling factor applied to all elements when accessed as doubles. Defaults to `1.0`. Setting this property updates the file header instantly.
*   **seed** (*int*): The random seed used to generate the matrix, if applicable. Read-only. Returns `0` if no seed was recorded.
*   **is_temporary** (*bool*): Indicates whether the backing storage is temporary (RAM or temp file) and should be deleted/released on exit.
*   **shape** (*tuple[int, int]*): The dimensions of the matrix `(rows, cols)`.
*   **backing_file** (*str*): Absolute path to the backing file on disk.
*   **properties** (*MutableMapping[str, Any]*): Semantic properties and cached-derived values.

	- Gospel assertions are authoritative (not truth-validated).
	- Boolean-like keys use tri-state semantics: unset means the key is absent.
	- Incompatible asserted states raise immediately (no payload scan).

	See [[guides/release1/properties.md|R1 Properties]] and [[guides/Storage and Memory.md|Storage and Memory]].

    **Internal routing note**: PyCauset mirrors boolean properties into a compact C++ bitmask for fast dispatch decisions. The mirror is updated whenever `properties` is mutated, but the bitmask is not part of the public API.

## Methods

### `rows()` / `cols()` / `size()`

*   `rows()` and `cols()` report the logical shape.
*   `size()` reports the total element count.

### `close()`
Releases the memory-mapped file handle or frees the RAM buffer. The matrix object becomes unusable after calling this method.

### `fill(value)`

Fill the matrix with a scalar value.

This is an explicit full write. On very large disk-backed matrices, this can be a long I/O operation.

### `get_backing_file()`
Returns the absolute path to the backing file on disk.

### `get_element_as_double(i, j)`
Returns the element at $(i, j)$ as a double-precision float, multiplied by `scalar`.

### `trace()`
Returns the trace of the matrix (sum of diagonal elements).

For rectangular matrices, this uses the diagonal length $\min(\text{rows}, \text{cols})$.
*   **Caching**: The result is cached in memory. Subsequent calls return the cached value instantly.
*   **Persistence**: When the matrix is saved using `save()`, cached-derived values may be written to the file’s typed metadata and automatically restored upon loading.

### `determinant()`
Returns the determinant of the matrix.

This is a **square-only** operation and will raise for non-square shapes.
*   **Caching**: The result is cached in memory.
*   **Persistence**: May be saved to typed metadata and restored upon loading.

### `transpose()` / `T`
Returns a transposed view of the same underlying storage (usually metadata-only; no element-wise copy).

### Indexing (`M[i, j]`)
Element access uses NumPy-style indexing:

*   Read: `x = M[i, j]`
*   Write: `M[i, j] = value`

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
*   [[project/protocols/NumPy Alignment Protocol.md|NumPy Alignment Protocol]]
