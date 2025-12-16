# pycauset.TriangularFloatMatrix

A memory-mapped upper triangular matrix storing 64-bit floating point numbers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.TriangularFloatMatrix(n: int, has_diagonal: bool = False)
```

Creates a new triangular matrix of size `n x n`.

*   `n`: The number of rows/columns.
*   `has_diagonal`: If `True`, the matrix stores and allows access to the diagonal elements. If `False` (default), the matrix is strictly upper triangular (diagonal is implicitly 0).

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### `get(i: int, j: int) -> float`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: float)`
Set the value at row `i` and column `j`.

### `multiply(other: TriangularFloatMatrix) -> TriangularFloatMatrix`
Multiply this matrix by another `TriangularFloatMatrix`.

### `invert() -> TriangularFloatMatrix`
Compute the inverse of the matrix.

### `__getitem__(idx: tuple) -> float`
Get element using `[i, j]` syntax.

### `__setitem__(idx: tuple, value: float)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.
