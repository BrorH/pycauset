```
pycauset.Matrix(source, saveas=None, populate=True)
```
The base class for all matrix types in pycauset. Also serves as a concrete implementation for dense, in-memory matrices.

All matrix types ([[pycauset.CausalMatrix]], [[pycauset.IntegerMatrix]], [[pycauset.FloatMatrix]]) inherit from this class and share common functionality like printing and shape properties.

### Parameters:
- source: Either an `int` dimension, a nested list/tuple describing an $N \times N$ matrix, or a NumPy array.
- saveas: str (_optional_). **Warning**: `pycauset.Matrix` does not support disk storage. This parameter is ignored with a warning. Use [[pycauset.CausalMatrix]] for disk-backed storage.
- populate: bool (_optional_). Only used when `source` is an `int`. If `True`, the matrix is filled with random bits using the current [[pycauset.seed]]; otherwise it starts empty.

### Returns:
[[pycauset.Matrix]] instance

