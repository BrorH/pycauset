# pycauset.TriangularIntegerMatrix

A memory-mapped upper triangular matrix storing 32-bit integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.TriangularIntegerMatrix(n: int, has_diagonal: bool = False)
```

Creates a new triangular matrix of size `n x n`.

*   `n`: The number of rows/columns.
*   `has_diagonal`: If `True`, the matrix stores and allows access to the diagonal elements. If `False` (default), the matrix is strictly upper triangular (diagonal is implicitly 0).

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### Indexing

Element access uses NumPy-style indexing:

```python
x = M[i, j]
M[i, j] = value
```

### `invert() -> TriangularIntegerMatrix`
Compute the inverse of the matrix.

### `__invert__() -> TriangularIntegerMatrix`
Compute the bitwise NOT of the matrix.


### `__repr__() -> str`
String representation of the matrix.
