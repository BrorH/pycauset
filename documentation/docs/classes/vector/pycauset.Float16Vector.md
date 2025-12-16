````markdown
# pycauset.Float16Vector

A memory-mapped vector storing 16-bit floating point numbers (half precision). Inherits from [VectorBase](pycauset.VectorBase.md).

## Constructor

```python
pycauset.Float16Vector(n: int)
```

## Notes

Half precision is primarily a storage and bandwidth optimization. Supported operations are gated by the support matrix (see `documentation/internals/DType System.md`).

````
