# pycauset.SymmetricMatrix

```python
class pycauset.SymmetricMatrix(TriangularMatrix)
```

A matrix where $A_{ij} = A_{ji}$.

This class stores only the upper triangular part of the matrix (including the diagonal) to save memory. It supports both symmetric and anti-symmetric matrices.

## Parameters

*   **n** (int): The number of rows/columns (matrix is square).

Note: For anti-symmetric matrices, use [[docs/classes/AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]] instead.

## Methods

### Indexing

Element access uses NumPy-style indexing:

*   Read: `x = A[i, j]` (if `i > j`, reads from the stored upper triangle)
*   Write: `A[i, j] = value` (if `i > j`, writes into the stored upper triangle)

Note: For anti-symmetric matrices, the diagonal must be zero.

### `transpose()`
Returns a new matrix representing the transpose.
*   For symmetric matrices, this returns a copy of itself.

For anti-symmetric matrices, see [[docs/classes/AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]].

## Storage
Uses a packed upper-triangular format. The storage size is approximately $N(N+1)/2$ elements.
