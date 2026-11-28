# pycauset.to_numpy

```python
pycauset.to_numpy(obj, *, allow_huge=False, dtype=None, copy=True)
```

Convert a PyCauset object (matrix/vector/block matrix) to a NumPy array.

This is the **explicit** NumPy export entrypoint. It exists because converting out-of-core objects to NumPy can cause surprise full materialization; `to_numpy` enforces the same safety rules described in [[guides/Numpy Integration.md|NumPy Integration]] and [[guides/Storage and Memory.md|Storage and Memory]].

## Parameters

- **obj**: A PyCauset matrix/vector (and selected internal matrix-like objects).
- **allow_huge** (bool, default `False`):
  - When `False`, the size guard applies: an export that would not fit in free RAM, or exceeds the
    `set_export_max_bytes` ceiling, raises. RAM- and file-backed objects are treated alike.
  - Set to `True` only when you intentionally want to materialize into RAM.
- **dtype** (optional): the NumPy dtype of the result (converted after the export; the size
  guard counts both arrays).
- **copy** (bool, default `True`): every export is a new array. `copy=False` raises
  `ValueError`, as `np.asarray(x, copy=False)` does when a copy cannot be avoided.

## Returns

- A `numpy.ndarray`.

## Exceptions

- `RuntimeError` if NumPy is unavailable.
- `RuntimeError` if the size guard blocks the export (too large, without `allow_huge=True`).
- `ValueError` for `copy=False`.

## Examples

```python
import pycauset as pc
import numpy as np

M = pc.zeros((3, 3), dtype="float32")
arr = pc.to_numpy(M)
assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
wide = pc.to_numpy(M, dtype=np.float64)
```

## See also

- [[docs/functions/pycauset.set_export_max_bytes.md|pycauset.set_export_max_bytes]]
- [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]]
- [[guides/Numpy Integration.md|NumPy Integration]]
- [[guides/Storage and Memory.md|Storage and Memory]]
