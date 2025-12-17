# pycauset.IntegerMatrix

A memory-mapped dense matrix storing 32-bit integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.IntegerMatrix(n: int)
pycauset.IntegerMatrix(rows: int, cols: int)
```

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### `get(i: int, j: int) -> int`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: int)`
Set the value at row `i` and column `j`.

### `multiply(other: IntegerMatrix) -> IntegerMatrix`
Multiply this matrix by another `IntegerMatrix`.

### `__getitem__(idx: tuple) -> int`
Get element using `[i, j]` syntax.

### `__setitem__(idx: tuple, value: int)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[guides/Matrix Guide|Matrix Guide]]
