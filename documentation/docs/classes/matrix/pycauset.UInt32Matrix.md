# pycauset.UInt32Matrix

A memory-mapped dense matrix storing 32-bit unsigned integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.UInt32Matrix(n: int)
pycauset.UInt32Matrix(rows: int, cols: int)
pycauset.UInt32Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `uint32`.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

Inherits the common matrix interface from [[pycauset.MatrixBase]].
