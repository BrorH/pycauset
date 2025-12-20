# pycauset.UInt16Matrix

A memory-mapped dense matrix storing 16-bit unsigned integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.UInt16Matrix(n: int)
pycauset.UInt16Matrix(rows: int, cols: int)
pycauset.UInt16Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `uint16`.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

Inherits the common matrix interface from [[pycauset.MatrixBase]].
