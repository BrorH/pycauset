# pycauset.TriangularIntegerMatrix

A memory-mapped upper triangular matrix storing 32-bit integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.TriangularIntegerMatrix(n: int, backing_file: str = "", has_diagonal: bool = False)
```

Creates a new triangular matrix of size `n x n`.

*   `n`: The number of rows/columns.
*   `backing_file`: Optional path to the file where data will be stored. If empty, a temporary file is created.
*   `has_diagonal`: If `True`, the matrix stores and allows access to the diagonal elements. If `False` (default), the matrix is strictly upper triangular (diagonal is implicitly 0).

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### `get(i: int, j: int) -> int`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: int)`
Set the value at row `i` and column `j`.

### `invert() -> TriangularIntegerMatrix`
Compute the inverse of the matrix.

### `__invert__() -> TriangularIntegerMatrix`
Compute the bitwise NOT of the matrix.

### `__getitem__(idx: tuple) -> float`
Get element using `[i, j]` syntax. Returns float for compatibility.

### `__setitem__(idx: tuple, value: int)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.
