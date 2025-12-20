# pycauset.IntegerVector

A memory-mapped vector storing 32-bit integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.IntegerVector(n: int)
pycauset.IntegerVector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `int32`.

## Properties

### `shape`
Returns a tuple representing the dimensions of the vector.

## Methods

### `__getitem__(i: int) -> int`
Get the value at index `i`.

### `__setitem__(i: int, value: int)`
Set the value at index `i`.

### `__len__() -> int`
Get the size of the vector.

### `__repr__() -> str`
String representation of the vector.
