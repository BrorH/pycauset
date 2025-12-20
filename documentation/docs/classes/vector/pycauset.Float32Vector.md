# pycauset.Float32Vector

A memory-mapped vector storing 32-bit floating point numbers (single precision). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.Float32Vector(n: int)
pycauset.Float32Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `float32`.

## Methods

Inherits the common vector interface from [[pycauset.VectorBase]].
