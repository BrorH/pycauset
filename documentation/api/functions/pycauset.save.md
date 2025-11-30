# pycauset.save

```python
pycauset.save(obj: PersistentObject, path: str)
```

Saves a persistent object (matrix or vector) to a permanent location on disk.

This function attempts to create a hard link to the object's backing file to avoid data duplication. If a hard link cannot be created (e.g., across different filesystems), it falls back to copying the file.

## Parameters

*   **obj** (*PersistentObject*): The object to save.
*   **path** (*str*): The destination path where the object should be saved.

## Example

```python
C = pycauset.CausalMatrix(1000)
pycauset.save(C, "my_saved_matrix.pycauset")
```
