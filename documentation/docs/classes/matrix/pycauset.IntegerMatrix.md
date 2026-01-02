# pycauset.IntegerMatrix

A memory-mapped dense matrix storing 32-bit integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.IntegerMatrix(n: int)
pycauset.IntegerMatrix(rows: int, cols: int)
pycauset.IntegerMatrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `int32`.

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

### `multiply(other: IntegerMatrix) -> IntegerMatrix`
Multiply this matrix by another `IntegerMatrix`.


### `__repr__() -> str`
String representation of the matrix.

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[guides/Matrix Guide|Matrix Guide]]
