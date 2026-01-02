# pycauset.Float32Matrix

```python
class Float32Matrix(MatrixBase)
```

A dense matrix storing 32-bit floating point numbers (`float`).

## Overview

This class is functionally identical to [[pycauset.FloatMatrix]] (which uses 64-bit doubles) but uses **half the storage**.

To allocate a `Float32Matrix`, use `pycauset.empty((rows, cols), dtype="float32")` or `pycauset.zeros((rows, cols), dtype="float32")`.

**Use this class for large matrices** where memory/disk I/O is the bottleneck and extreme precision is not required.

**GPU Acceleration:**
Matrix multiplication (`multiply` or `@`) is **GPU-accelerated** for `Float32Matrix`. It is significantly faster than `FloatMatrix` (Double Precision) on consumer GPUs (e.g., GeForce series) which often have much higher FP32 throughput than FP64.


## Constructor

```python
pycauset.Float32Matrix(n: int)
pycauset.Float32Matrix(rows: int, cols: int)
pycauset.Float32Matrix(array: numpy.ndarray)
```

When constructed from a NumPy array, the array must be rank-2 with dtype `float32`.

*   `n`: The size of a square matrix ($N \times N$).
*   `rows`, `cols`: The shape of a rectangular matrix.

## Methods

Inherits all methods from [[pycauset.MatrixBase]].

*   Indexing: read with `M[i, j]`, write with `M[i, j] = value`.
*   `inverse()` / `invert()`: Computes the inverse (**square-only**; requires `rows == cols`).

## See Also

*   [[pycauset.FloatMatrix]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.zeros.md|pycauset.zeros]]
*   [[guides/Matrix Guide|Matrix Guide]]

