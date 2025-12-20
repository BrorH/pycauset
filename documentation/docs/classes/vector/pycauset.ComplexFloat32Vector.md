# pycauset.ComplexFloat32Vector

A memory-mapped vector storing complex numbers in `complex_float32` format (equivalent to NumPy `complex64`). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.ComplexFloat32Vector(n: int)
pycauset.ComplexFloat32Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `complex64`.

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
