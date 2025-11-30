# pycauset.DenseBitMatrix

A memory-mapped dense matrix storing boolean values (bits). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.DenseBitMatrix(n: int, backing_file: str = "")
```

## Static Methods

### `random(n: int, density: float = 0.5, backing_file: str = "", seed: int = None) -> DenseBitMatrix`
Create a random dense bit matrix.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### `get(i: int, j: int) -> bool`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: bool)`
Set the value at row `i` and column `j`.

### `multiply(other: DenseBitMatrix, saveas: str = "") -> DenseBitMatrix`
Multiply this matrix by another `DenseBitMatrix`.

### `__invert__() -> DenseBitMatrix`
Compute the bitwise NOT of the matrix.

### `__getitem__(idx: tuple) -> bool`
Get element using `[i, j]` syntax.

### `__setitem__(idx: tuple, value: bool)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.
