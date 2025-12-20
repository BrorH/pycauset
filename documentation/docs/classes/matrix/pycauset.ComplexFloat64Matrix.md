# pycauset.ComplexFloat64Matrix

A memory-mapped dense matrix storing complex numbers in `complex_float64` format (equivalent to NumPy `complex128`). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.ComplexFloat64Matrix(n: int)
pycauset.ComplexFloat64Matrix(rows: int, cols: int)
pycauset.ComplexFloat64Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `complex128`.

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
