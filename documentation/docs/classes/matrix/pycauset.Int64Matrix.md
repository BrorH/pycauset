# pycauset.Int64Matrix

A memory-mapped dense matrix storing 64-bit signed integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.Int64Matrix(n: int)
pycauset.Int64Matrix(rows: int, cols: int)
pycauset.Int64Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `int64`.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

Inherits the common matrix interface from [[pycauset.MatrixBase]].
