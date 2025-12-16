````markdown
# pycauset.ComplexFloat32Matrix

A memory-mapped dense matrix storing complex numbers in `complex_float32` format (equivalent to NumPy `complex64`). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.ComplexFloat32Matrix(n: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).

````
