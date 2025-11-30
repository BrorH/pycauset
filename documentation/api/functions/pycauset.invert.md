# pycauset.invert

```python
pycauset.invert(matrix: Any) -> Any
```

Compute the linear algebra inverse ($A^{-1}$) of a matrix.

## Parameters

*   **matrix** (*MatrixBase* or *array-like*): The matrix to invert.

## Returns

*   **MatrixBase** or *array-like*: The inverse matrix.

## Raises

*   **RuntimeError**: If the matrix is singular (e.g., strictly upper triangular matrices like `TriangularBitMatrix`).
*   **TypeError**: If the object does not support inversion.
