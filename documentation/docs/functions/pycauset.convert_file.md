# pycauset.convert_file

```python
pycauset.convert_file(src_path, dst_path, *, dst_format=None, allow_huge=False, dtype=None, npz_key=None)
```

Convert between PyCauset snapshots and NumPy container formats.

- Supported formats: `.pycauset` (canonical snapshot), `.npy`, `.npz`.
- `dst_format` defaults from `dst_path` when omitted; must be one of the supported suffixes.
- `.npz` imports default to the alphabetically first key; set `npz_key` to choose a specific array name.
- Exports honor the NumPy size guard (see [[docs/functions/pycauset.to_numpy.md|to_numpy]]); `allow_huge=True` skips it.
- Optional `dtype` casts on export (to NumPy formats) before writing.

## Exceptions / warnings

- `ValueError` if source or destination format is not one of `.pycauset`, `.npy`, `.npz` (or cannot be inferred from the suffix).
- `RuntimeError` if NumPy is unavailable when exporting to `.npy`/`.npz`.
- Size guard: an export to a NumPy format that would not fit in free RAM (or exceeds the `set_export_max_bytes` ceiling) raises unless `allow_huge=True`.

## Parameters

- **src_path** (str | Path): Source file path; suffix must be `.pycauset`, `.npy`, or `.npz`.
- **dst_path** (str | Path): Destination file path; suffix or `dst_format` selects the output format.
- **dst_format** (str, optional): Override destination format (`"pycauset"`, `"npy"`, or `"npz"`). If omitted, inferred from `dst_path`.
- **allow_huge** (bool, default `False`): Skips the size guard.
- **dtype** (optional): Override dtype on export to NumPy formats.
- **npz_key** (str, optional): Key to read from/write to when the source or destination is `.npz`. Defaults to the alphabetically first key on import and to `"array"` on export.

## Returns

- **Path**: The destination path.

## Examples

```python
import numpy as np
import pycauset as pc

pc.save(pc.matrix([[1.0, 2.0], [3.0, 4.0]]), "A.pycauset")
# Snapshot (.pycauset) -> NumPy .npy -> snapshot
pc.convert_file("A.pycauset", "A.npy")
pc.convert_file("A.npy", "A_roundtrip.pycauset")

# Extract from an npz archive into a snapshot
np.savez("bundle.npz", vector0=np.arange(4.0))
pc.convert_file("bundle.npz", "vec.pycauset", npz_key="vector0")

# Export to npz with explicit dtype and large-export opt-in
pc.save(pc.zeros((4, 4)), "big.pycauset")
pc.convert_file("big.pycauset", "big.npz", allow_huge=True, dtype="float32", npz_key="arr")
```

## Future format targets (not implemented yet)

These are under consideration for later releases; they are not supported by `convert_file` today:

- MatrixMarket `.mtx` (sparse/text interchange)
- MATLAB `.mat` (engineering/scientific interop)
- Parquet / Arrow / CSV (tabular pipelines; CSV mainly for debugging)
- HDF5/NetCDF (only if a low-maintenance reader fits the budget)

## See Also

- [[docs/functions/pycauset.save.md|pycauset.save]]
- [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Storage and Memory|Storage and Memory]]
- [[guides/Numpy Integration.md|NumPy Integration]]
- NumPy helpers in the Python API: `load_npy`, `load_npz`, `save_npy`, `save_npz`
