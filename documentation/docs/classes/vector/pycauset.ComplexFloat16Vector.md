# pycauset.ComplexFloat16Vector

A memory-mapped vector storing complex numbers in `complex_float16` format (two-plane float16: real + imaginary). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.ComplexFloat16Vector(n: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
