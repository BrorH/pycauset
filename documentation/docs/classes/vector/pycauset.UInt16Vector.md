# pycauset.UInt16Vector

A memory-mapped vector storing 16-bit unsigned integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.UInt16Vector(n: int)
pycauset.UInt16Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `uint16`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
