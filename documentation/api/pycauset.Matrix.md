# pycauset.Matrix

```python
pycauset.Matrix(source)
```

The base class for all matrix types in pycauset. Also serves as a concrete implementation for dense, in-memory matrices.

All matrix types ([[pycauset.CausalMatrix]], [[pycauset.IntegerMatrix]], [[pycauset.FloatMatrix]]) inherit from this class and share common functionality like printing and shape properties.

## Parameters

*   **source** (*int, list, or numpy.ndarray*): Either an `int` dimension, a nested list/tuple describing an $N \times N$ matrix, or a NumPy array.

## Returns

*   **Matrix**: A new instance of [[pycauset.Matrix]].

