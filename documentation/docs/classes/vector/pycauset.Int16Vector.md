# pycauset.Int16Vector

A memory-mapped vector storing 16-bit signed integers. Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.Int16Vector(n: int)
pycauset.Int16Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `int16`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
