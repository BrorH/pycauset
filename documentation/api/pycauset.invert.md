# pycauset.invert

```python
pycauset.invert(matrix)
```

Computes the linear algebra inverse ($A^{-1}$) of a matrix.

*   **Note**: Strictly upper triangular matrices (such as `TriangularBitMatrix`, `IntegerMatrix`, and `TriangularFloatMatrix`) are **singular** (determinant is 0) and cannot be inverted. Calling this function on such matrices will raise a `RuntimeError`.

## Parameters

*   **matrix** (*MatrixBase*): The matrix to invert.

## Returns

*   **MatrixBase**: The inverse matrix.

## Raises

*   **RuntimeError**: If the matrix is singular or if inversion is not implemented for the matrix type.

## Dense Matrix Inversion

The `FloatMatrix` class supports general matrix inversion using Gaussian elimination with partial pivoting. This allows you to invert dense matrices, provided they are non-singular (determinant is non-zero).

```python
import pycauset

# Create a dense FloatMatrix
# [[4, 7],
#  [2, 6]]
m = pycauset.FloatMatrix(2)
m[0, 0] = 4.0
m[0, 1] = 7.0
m[1, 0] = 2.0
m[1, 1] = 6.0

# Compute Inverse
# [[ 0.6, -0.7],
#  [-0.2,  0.4]]
inv = m.invert()
# OR
inv = pycauset.invert(m)

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
