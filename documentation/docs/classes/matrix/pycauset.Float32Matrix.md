# pycauset.Float32Matrix

```python
class Float32Matrix(MatrixBase)
```

A dense matrix storing 32-bit floating point numbers (`float`).

## Overview

This class is functionally identical to [[pycauset.FloatMatrix]] (which uses 64-bit doubles) but uses **half the storage**. It is the default matrix type chosen by the [[pycauset.Matrix]] factory for sufficiently large matrices (currently $N \ge 10,000$), unless you override precision.

**Use this class for large matrices** where memory/disk I/O is the bottleneck and extreme precision is not required.

**GPU Acceleration:**
Matrix multiplication (`multiply` or `@`) is **GPU-accelerated** for `Float32Matrix`. It is significantly faster than `FloatMatrix` (Double Precision) on consumer GPUs (e.g., GeForce series) which often have much higher FP32 throughput than FP64.


## Constructor

```python
pycauset.Float32Matrix(n: int)
```

*   `n`: The size of the matrix ($N \times N$).

## Methods

Inherits all methods from [[pycauset.MatrixBase]].

*   `get(i, j)`: Returns the element at $(i, j)$ as a float.
*   `set(i, j, value)`: Sets the element at $(i, j)$.
*   `eigenvalues()`: Computes eigenvalues.
*   `inverse()` / `invert()`: Computes the inverse.

## See Also

*   [[pycauset.FloatMatrix]]

