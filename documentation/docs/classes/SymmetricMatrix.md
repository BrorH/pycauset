# SymmetricMatrix

The `SymmetricMatrix` class represents a square matrix $A$ where $A_{ij} = A_{ji}$.

It is optimized for storage efficiency, storing only the upper triangular part (including the diagonal) in memory. This reduces memory usage by approximately 50% compared to a dense matrix.

## Class Hierarchy

*   Inherits from: `TriangularMatrixBase` -> `MatrixBase` -> `PersistentObject`

## Constructors

### `SymmetricMatrix(n, scalar=0.0)`

Creates a new symmetric matrix.

*   **n** (int): The number of rows/columns (matrix is $N \times N$).
*   **scalar** (float or complex, optional): A scalar multiplier associated with the matrix. Defaults to 0.0.

### `from_triangular(source)`

Static method to create a SymmetricMatrix from a TriangularMatrix.
It copies the upper triangular part (including diagonal) of the source matrix.
The resulting matrix $S$ satisfies $S_{ij} = S_{ji} = \text{source}_{ij}$ for $i \le j$.

*   **source** (TriangularMatrix): The source matrix.

## Properties

### `is_antisymmetric`
*   **Type**: `bool`
*   **Description**: Returns `False`.

### `shape`
*   **Type**: `tuple`
*   **Description**: Returns `(n, n)`.

### `T`
*   **Type**: `SymmetricMatrix`
*   **Description**: Returns the transpose (which is a copy of itself for symmetric matrices).

## Methods

### Indexing
Read elements using NumPy-style indexing: `x = S[i, j]`.
*   **i, j** (int): Indices.
*   **Returns**: The value at $(i, j)$.
    *   If $i > j$, it accesses the stored value at $(j, i)$.

Write elements using NumPy-style indexing: `S[i, j] = value`.
*   **i, j** (int): Indices.
*   **value**: The value to set.
    *   If $i > j$, it sets the stored value at $(j, i)$.

### `copy()`
Creates a copy of the matrix.

### `close()`
Closes the memory map and releases resources.

## Usage Example

```python
import pycauset as pc

# Create a Symmetric Matrix
S = pc.SymmetricMatrix(100)
S[10, 5] = 3.14
print(S[5, 10])  # Output: 3.14
```
