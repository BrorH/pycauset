# pycauset.set_export_max_bytes

```python
pycauset.set_export_max_bytes(limit)
```

Set a global materialization ceiling (in bytes) for exports to NumPy.

This controls when `np.asarray(obj)`, `np.array(obj)`, and [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]] are allowed to create a dense in-RAM NumPy array.

Notes:

- The guard is by size only: RAM-backed and file-backed objects are treated alike.
- With `None` (the default) an export raises only when the new array would not fit in free RAM.
- `pycauset.to_numpy(obj, allow_huge=True)` skips the check.

## Parameters

- **limit** (int | None): Maximum allowed dense export size (bytes). Use `None` to disable.

## Examples

```python
import pycauset as pc

# Allow up to 512MB NumPy exports
pc.set_export_max_bytes(512 * 1024 * 1024)

# Back to the default: only exports that would not fit in free RAM raise
pc.set_export_max_bytes(None)
```

## See also

- [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[guides/Numpy Integration.md|NumPy Integration]]
