# pycauset.save_npy

```python
pycauset.save_npy(obj, path, *, allow_huge=False, dtype=None)
```

Saves a PyCauset matrix or vector to a NumPy `.npy` file.

## Parameters

*   **obj** (matrix or vector): The object to save.
*   **path** (str or Path): Destination `.npy` path.
*   **allow_huge** (bool): If `True`, bypasses the NumPy materialization guard.
*   **dtype** (optional): NumPy dtype for the export.

## Returns

The resolved `Path` of the written file.

## See also

* [[docs/functions/pycauset.save_npz.md|pycauset.save_npz]]
* [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]]
