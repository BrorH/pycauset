# pycauset.save

```python
pycauset.save(obj: Union[PersistentObject, CausalSet], path: str)
```

Saves a persistent object (matrix, vector) or a `CausalSet` to a permanent location on disk.

*   **For Matrices/Vectors**: Attempts to create a hard link to the backing file. If that fails, it copies the file.
*   **For CausalSets**: Delegates to `CausalSet.save()`, creating a `.causet` ZIP archive containing metadata and the matrix.

## Parameters

*   **obj** (*PersistentObject* | *CausalSet*): The object to save.
*   **path** (*str*): The destination path.

## Example

```python
# Save a raw matrix
pc.save(matrix, "data.pycauset")

# Save a Causal Set
pc.save(causet, "universe.causet")
```

## See Also

*   [[pycauset.CausalSet.save]]

