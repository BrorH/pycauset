# pycauset.UInt64Matrix

A memory-mapped dense matrix storing 64-bit unsigned integers. Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.UInt64Matrix(n: int)
pycauset.UInt64Matrix(rows: int, cols: int)
pycauset.UInt64Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `uint64`.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

Inherits the common matrix interface from [[pycauset.MatrixBase]].
