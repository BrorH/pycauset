# pycauset.FloatMatrix

A memory-mapped dense matrix storing 64-bit floating point numbers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.FloatMatrix(n: int)
pycauset.FloatMatrix(rows: int, cols: int)
pycauset.FloatMatrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `float64`.

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

### `multiply(other: FloatMatrix) -> FloatMatrix`
Multiply this matrix by another `FloatMatrix`.

### `invert() -> FloatMatrix`
Compute the inverse of the matrix.

This is a **square-only** operation and will raise for non-square shapes.

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[guides/Matrix Guide|Matrix Guide]]
