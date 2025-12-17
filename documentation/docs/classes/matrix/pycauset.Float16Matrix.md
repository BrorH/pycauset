# pycauset.Float16Matrix

A memory-mapped dense matrix storing 16-bit floating point numbers (half precision). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.Float16Matrix(n: int)
pycauset.Float16Matrix(rows: int, cols: int)
```

## Notes

Half precision is primarily a storage and bandwidth optimization. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).
