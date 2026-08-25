# pycauset.load_npz

```python
pycauset.load_npz(path, *, key=None)
```

Loads a NumPy `.npz` archive into a PyCauset matrix or vector.

## Parameters

*   **path** (str or Path): Path to the `.npz` archive.
*   **key** (str, optional): Which array to load. Defaults to the first key in the archive.

## Returns

A PyCauset matrix (2D array) or vector (1D array).

## See also

* [[docs/functions/pycauset.load_npy.md|pycauset.load_npy]]
* [[docs/functions/pycauset.save_npz.md|pycauset.save_npz]]
