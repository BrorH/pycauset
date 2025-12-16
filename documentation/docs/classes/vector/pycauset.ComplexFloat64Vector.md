````markdown
# pycauset.ComplexFloat64Vector

A memory-mapped vector storing complex numbers in `complex_float64` format (equivalent to NumPy `complex128`). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.ComplexFloat64Vector(n: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).

````
