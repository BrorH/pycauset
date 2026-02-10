# pycauset.cuda.disable

```python
pycauset.cuda.disable()
```

Disables the CUDA backend for the current session.

After disabling, the dispatcher routes operations to CPU-only implementations
until `pycauset.cuda.enable()` succeeds again.

## Returns

*   **None**

## Behavior

*   Turns off GPU routing for new operations.
*   Does **not** invalidate or mutate existing matrices.
*   Does **not** delete the cached hardware profile on disk.

## Exceptions / warnings

*   This does not unload already-created matrices; it only affects future routing.

## Example

```python
import pycauset as pc

pc.cuda.disable()

# Switch back to auto routing later
pc.cuda.enable()
pc.cuda.force_backend("auto")
```

## See also

- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.is_available.md|pycauset.cuda.is_available]]
- [[docs/functions/pycauset.cuda.force_backend.md|pycauset.cuda.force_backend]]
