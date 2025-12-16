# pycauset.BitVector

A memory-mapped vector storing boolean values (bits). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.BitVector(n: int)
```

## Properties

### `shape`
Returns a tuple representing the dimensions of the vector.

## Methods

### `__getitem__(i: int) -> bool`
Get the value at index `i`.

### `__setitem__(i: int, value: bool)`
Set the value at index `i`.

### `__len__() -> int`
Get the size of the vector.

### `__repr__() -> str`
String representation of the vector.
