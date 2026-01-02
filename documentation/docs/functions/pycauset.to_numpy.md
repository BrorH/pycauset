# pycauset.to_numpy

```python
pycauset.to_numpy(obj, *, allow_huge=False, dtype=None, copy=True)
```

Convert a PyCauset object (matrix/vector/block matrix) to a NumPy array.

This is the **explicit** NumPy export entrypoint. It exists because converting out-of-core objects to NumPy can cause surprise full materialization; `to_numpy` enforces the same safety rules described in [[guides/Numpy Integration.md|NumPy Integration]] and [[guides/Storage and Memory.md|Storage and Memory]].

## Parameters

- **obj**: A PyCauset matrix/vector (and selected internal matrix-like objects).
- **allow_huge** (bool, default `False`):
  - When `False`, exporting spill/file-backed objects hard-errors to prevent surprise full materialization.
  - Set to `True` only when you intentionally want to materialize into RAM.
- **dtype** (optional): Override NumPy dtype on export.
- **copy** (bool, default `True`): Return a NumPy array whose buffer is independent of the PyCauset object.

## Returns

- A `numpy.ndarray`.

## Exceptions

- `RuntimeError` if NumPy is unavailable.
- `RuntimeError` if export is blocked by the materialization guard (file-backed / too-large without opt-in).

## Examples

```python
import pycauset as pc
import numpy as np

M = pc.zeros((3, 3), dtype="float32")
arr = pc.to_numpy(M)
assert isinstance(arr, np.ndarray)
```

## See also

- [[docs/functions/pycauset.set_export_max_bytes.md|pycauset.set_export_max_bytes]]
- [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]]
- [[guides/Numpy Integration.md|NumPy Integration]]
- [[guides/Storage and Memory.md|Storage and Memory]]
