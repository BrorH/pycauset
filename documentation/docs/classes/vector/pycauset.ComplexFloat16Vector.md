# pycauset.ComplexFloat16Vector

A memory-mapped vector storing complex numbers in `complex_float16` format (two-plane float16: real + imaginary). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.ComplexFloat16Vector(n: int)
pycauset.ComplexFloat16Vector(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-1 with dtype `complex64` or `complex128`.

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
