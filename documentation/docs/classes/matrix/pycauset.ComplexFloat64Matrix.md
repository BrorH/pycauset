````markdown
# pycauset.ComplexFloat64Matrix

A memory-mapped dense matrix storing complex numbers in `complex_float64` format (equivalent to NumPy `complex128`). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.ComplexFloat64Matrix(n: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).

````
