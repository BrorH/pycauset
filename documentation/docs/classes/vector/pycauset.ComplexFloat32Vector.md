````markdown
# pycauset.ComplexFloat32Vector

A memory-mapped vector storing complex numbers in `complex_float32` format (equivalent to NumPy `complex64`). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.ComplexFloat32Vector(n: int)
```

## Notes

Complex support is limited to complex floats. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).

````
