# pycauset.save

```python
pycauset.save(matrix, path)
```

Saves a matrix to a permanent location on disk.

This function attempts to create a hard link to the matrix's backing file to avoid data duplication. If a hard link cannot be created (e.g., across different filesystems), it falls back to copying the file.

## Parameters

*   **matrix** (*MatrixBase*): The matrix object to save. Must be an instance of a file-backed matrix class (e.g., `TriangularBitMatrix`, `IntegerMatrix`).
*   **path** (*str* or *PathLike*): The destination path where the matrix should be saved.

## Example

```python
C = pycauset.CausalMatrix(1000)
pycauset.save(C, "my_saved_matrix.pycauset")
```
