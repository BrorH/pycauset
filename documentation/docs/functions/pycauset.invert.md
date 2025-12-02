# Matrix Inversion (Linear Algebra)

PyCauset provides an interface for computing the mathematical inverse ($A^{-1}$) of a matrix.

## Usage

You can use the `.invert()` method on a matrix object or the [[pycauset.invert]] function.

```python
import pycauset as pc

# Create a matrix (must be invertible)
# Note: TriangularBitMatrix and IntegerMatrix are strictly upper triangular
# and therefore singular (determinant is 0). They cannot be inverted.

try:
    m = pc.TriangularBitMatrix(5)
    inv = m.invert()
    # OR
    inv = pc.invert(m)
except RuntimeError as e:
    print(f"Inversion failed: {e}")
```

## Singularity of Triangular Matrices

The triangular matrix types in PyCauset ([[pycauset.TriangularBitMatrix]], [[pycauset.TriangularFloatMatrix]]) are **strictly upper triangular**. 

A strictly upper triangular matrix has zeros on the main diagonal. The determinant of a triangular matrix is the product of its diagonal entries. Therefore, the determinant of any strictly upper triangular matrix is 0, making it **singular** (non-invertible).

Attempting to invert these matrices will raise a `RuntimeError`.

## Dense Matrix Inversion

The [[pycauset.FloatMatrix]] class supports general matrix inversion using Gaussian elimination with partial pivoting. This allows you to invert dense matrices, provided they are non-singular (determinant is non-zero).

**Note**: [[pycauset.IntegerMatrix]] and [[pycauset.DenseBitMatrix]] are also dense, but direct inversion is not supported to avoid ambiguity (integer inversion usually results in floats). To invert them, convert them to [[pycauset.FloatMatrix]] first.

```python
import pycauset as pc

# Create a dense FloatMatrix
# [[4, 7],
#  [2, 6]]
m = pc.FloatMatrix(2)
m[0, 0] = 4.0
m[0, 1] = 7.0
m[1, 0] = 2.0
m[1, 1] = 6.0

# Compute Inverse
# [[ 0.6, -0.7],
#  [-0.2,  0.4]]
inv = m.invert()

print(inv[0, 0]) # 0.6
```

### Implementation Details

*   **Algorithm**: Gaussian elimination with partial pivoting.
*   **Parallelism**: The row operations are parallelized using OpenMP for performance on large matrices.
*   **Storage**: The operation creates a temporary backing file for the working matrix during computation, which is deleted automatically upon completion.
*   **Scalars**: If the input matrix has a scalar factor $S$, the resulting inverse will have a scalar factor $1/S$.

### Errors

A `RuntimeError` will be raised if:
*   The matrix is singular (determinant is 0).
*   The matrix is nearly singular (pivot element is close to zero, within a tolerance of 1e-12).

