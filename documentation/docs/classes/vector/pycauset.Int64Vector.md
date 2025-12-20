# pycauset.Int64Vector

A memory-mapped vector storing 64-bit signed integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.Int64Vector(n: int)
pycauset.Int64Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `int64`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
