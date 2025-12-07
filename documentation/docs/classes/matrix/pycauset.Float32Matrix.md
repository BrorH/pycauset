# pycauset.Float32Matrix

```python
class Float32Matrix(MatrixBase)
```

A dense matrix storing 32-bit floating point numbers (`float`).

## Overview

This class is functionally identical to [[pycauset.FloatMatrix]] (which uses 64-bit doubles) but uses **half the storage**. It is the default matrix type for matrices with $10,000 \le N < 100,000$.

**Use this class for large matrices** where memory/disk I/O is the bottleneck and extreme precision is not required.

**GPU Acceleration:**
Matrix multiplication (`multiply` or `@`) is **GPU-accelerated** for `Float32Matrix`. It is significantly faster than `FloatMatrix` (Double Precision) on consumer GPUs (e.g., GeForce series) which often have much higher FP32 throughput than FP64.


## Constructor

```python
pycauset.Float32Matrix(n: int, backing_file: str = "")
```

*   `n`: The size of the matrix ($N \times N$).
*   `backing_file`: (Optional) Path to the storage file. If omitted, a temporary file is used.

## Methods

Inherits all methods from [[pycauset.MatrixBase]].

*   `get(i, j)`: Returns the element at $(i, j)$ as a float.
*   `set(i, j, value)`: Sets the element at $(i, j)$.
*   `eigenvalues()`: Computes eigenvalues (promotes to double for calculation).
*   `eigenvectors()`: Computes eigenvectors.
*   `inverse()`: Computes the inverse.

## See Also

*   [[pycauset.FloatMatrix]]
*   [[pycauset.Float16Matrix]]
