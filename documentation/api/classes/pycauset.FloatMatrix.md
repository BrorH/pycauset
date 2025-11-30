# pycauset.FloatMatrix

A memory-mapped dense matrix storing 64-bit floating point numbers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.FloatMatrix(n: int, backing_file: str = "")
```

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### `get(i: int, j: int) -> float`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: float)`
Set the value at row `i` and column `j`.

### `multiply(other: FloatMatrix, saveas: str = "") -> FloatMatrix`
Multiply this matrix by another `FloatMatrix`.

### `invert() -> FloatMatrix`
Compute the inverse of the matrix.

### `__getitem__(idx: tuple) -> float`
Get element using `[i, j]` syntax.

### `__setitem__(idx: tuple, value: float)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.
