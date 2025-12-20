# pycauset.UInt32Vector

A memory-mapped vector storing 32-bit unsigned integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.UInt32Vector(n: int)
pycauset.UInt32Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `uint32`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
