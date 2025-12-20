# pycauset.Int8Matrix

A memory-mapped dense matrix storing 8-bit signed integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.Int8Matrix(n: int)
pycauset.Int8Matrix(rows: int, cols: int)
pycauset.Int8Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `int8`.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

Inherits the common matrix interface from [[pycauset.MatrixBase]].
