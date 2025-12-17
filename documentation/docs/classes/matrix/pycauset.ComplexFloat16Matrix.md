# pycauset.ComplexFloat16Matrix

A memory-mapped dense matrix storing complex numbers in `complex_float16` format (two-plane float16: real + imaginary). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.ComplexFloat16Matrix(n: int)
pycauset.ComplexFloat16Matrix(rows: int, cols: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
