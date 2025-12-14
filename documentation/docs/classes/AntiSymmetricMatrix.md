# AntiSymmetricMatrix

The `AntiSymmetricMatrix` class represents a square matrix $A$ where $A_{ij} = -A_{ji}$.

It is optimized for storage efficiency, storing only the upper triangular part in memory. This reduces memory usage by approximately 50% compared to a dense matrix.

## Class Hierarchy

*   Inherits from: `SymmetricMatrix` -> `TriangularMatrixBase` -> `MatrixBase` -> `PersistentObject`

## Constructors

### `AntiSymmetricMatrix(n, backing_file="", scalar=0.0)`

Creates a new anti-symmetric matrix.

*   **n** (int): The number of rows/columns (matrix is $N \times N$).
*   **backing_file** (str, optional): Path to a file for persistent storage. If empty, a temporary file is created.
*   **scalar** (float or complex, optional): A scalar multiplier associated with the matrix. Defaults to 0.0.

### `from_triangular(source, backing_file="")`

Static method to create an AntiSymmetricMatrix from a TriangularMatrix.
It copies the upper triangular part (excluding diagonal) of the source matrix.
The resulting matrix $A$ satisfies $A_{ij} = \text{source}_{ij}$ for $i < j$, $A_{ji} = -A_{ij}$, and $A_{ii} = 0$.

This is particularly useful for computing the Pauli-Jordan function $\Delta = K - K^T$ where $K$ is a triangular propagator.

*   **source** (TriangularMatrix): The source matrix.
*   **backing_file** (str, optional): Path to a file for persistent storage.

## Properties

### `is_antisymmetric`
*   **Type**: `bool`
*   **Description**: Always returns `True`.

### `shape`
*   **Type**: `tuple`
*   **Description**: Returns `(n, n)`.

### `T`
*   **Type**: `AntiSymmetricMatrix`
*   **Description**: Returns the transpose ($A^T = -A$).

## Methods

### `get(i, j)`
Returns the element at row `i` and column `j`.
*   **i, j** (int): Indices.
*   **Returns**: The value at $(i, j)$.
    *   If $i > j$, returns $-A_{ji}$.
    *   If $i == j$, returns 0.

### `set(i, j, value)`
Sets the element at row `i` and column `j`.
*   **i, j** (int): Indices.
*   **value**: The value to set.
    *   If $i > j$, it sets $A_{ji} = -value$.
    *   **Note**: The diagonal ($i=j$) must be 0. Attempting to set a non-zero diagonal value will raise an error.

### `copy()`
Creates a copy of the matrix.

### `close()`
Closes the memory map and releases resources.

## Usage Example

```python
import pycauset as pc

# Create an Anti-Symmetric Matrix (e.g., Pauli-Jordan Delta)
Delta = pc.AntiSymmetricFloat64Matrix(100)
Delta[10, 5] = 2.0
print(Delta[5, 10])  # Output: -2.0
print(Delta[5, 5])   # Output: 0.0
```
