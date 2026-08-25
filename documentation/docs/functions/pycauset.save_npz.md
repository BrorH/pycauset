# pycauset.save_npz

```python
pycauset.save_npz(obj, path, *, allow_huge=False, dtype=None, key="array")
```

Saves a PyCauset matrix or vector to a NumPy `.npz` archive.

## Parameters

*   **obj** (matrix or vector): The object to save.
*   **path** (str or Path): Destination `.npz` path.
*   **allow_huge** (bool): If `True`, bypasses the NumPy materialization guard.
*   **dtype** (optional): NumPy dtype for the export.
*   **key** (str): Archive key. Defaults to `"array"`.

## Returns

The resolved `Path` of the written file.

## See also

* [[docs/functions/pycauset.save_npy.md|pycauset.save_npy]]
* [[docs/functions/pycauset.load_npz.md|pycauset.load_npz]]
