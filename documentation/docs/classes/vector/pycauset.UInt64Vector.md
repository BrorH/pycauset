# pycauset.UInt64Vector

A memory-mapped vector storing 64-bit unsigned integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.UInt64Vector(n: int)
pycauset.UInt64Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `uint64`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
