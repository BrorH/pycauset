# pycauset.UInt8Vector

A memory-mapped vector storing 8-bit unsigned integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.UInt8Vector(n: int)
pycauset.UInt8Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `uint8`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
