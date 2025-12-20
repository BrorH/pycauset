# pycauset.Int8Vector

A memory-mapped vector storing 8-bit signed integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.Int8Vector(n: int)
pycauset.Int8Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `int8`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
