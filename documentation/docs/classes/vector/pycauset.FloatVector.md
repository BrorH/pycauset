# pycauset.FloatVector

A memory-mapped vector storing 64-bit floating point numbers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.FloatVector(n: int)
pycauset.FloatVector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `float64`.

## Properties

### `shape`
Returns a tuple representing the dimensions of the vector.

## Methods

### `__getitem__(i: int) -> float`
Get the value at index `i`.

### `__setitem__(i: int, value: float)`
Set the value at index `i`.

### `__len__() -> int`
Get the size of the vector.

### `__repr__() -> str`
String representation of the vector.
