# pycauset.Float16Matrix

```python
class Float16Matrix(MatrixBase)
```

A dense matrix storing 16-bit floating point numbers (Half Precision).

## Overview

This class uses **quarter the storage** of a standard [[pycauset.FloatMatrix]]. It is the default matrix type for massive matrices with $N \ge 100,000$.

**Use this class for massive scale simulations** where storage is the primary constraint.

### Precision Note
16-bit floats have limited precision (~3-4 decimal digits) and range ($\pm 65,504$). They are suitable for statistical properties of large causal sets (like spectral dimension) but not for precise numerical analysis.

## Constructor

```python
pycauset.Float16Matrix(n: int, backing_file: str = "")
```

*   `n`: The size of the matrix ($N \times N$).
*   `backing_file`: (Optional) Path to the storage file. If omitted, a temporary file is used.

## Methods

Inherits all methods from [[pycauset.MatrixBase]].

*   `get(i, j)`: Returns the element at $(i, j)$ as a float (converted from half).
*   `set(i, j, value)`: Sets the element at $(i, j)$.
*   `eigenvalues()`: Computes eigenvalues (promotes to double for calculation).
*   `eigenvectors()`: Computes eigenvectors.
*   `inverse()`: Computes the inverse.

## See Also

*   [[pycauset.FloatMatrix]]
*   [[pycauset.Float32Matrix]]
