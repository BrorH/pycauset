# pycauset.cuda.is_available

```python
pycauset.cuda.is_available() -> bool
```

Returns `True` if a CUDA device is loaded and active in the current session.

This is **not** a hardware probe. It reports whether the CUDA backend is
currently active (plugin loaded + device created). If you need to attempt
activation, call `pycauset.cuda.enable()` first.

## Returns

*   **bool**: Whether the CUDA backend is active.

## Notes

*   The CUDA plugin may auto-load at import time if it is available, in which
    case `is_available()` can return `True` without an explicit enable.
*   If the plugin cannot be loaded, this returns `False` and routing is CPU-only.

## Example

```python
import pycauset as pc

if pc.cuda.is_available():
    print("CUDA is active")
else:
    pc.cuda.enable()
    print("CUDA active:", pc.cuda.is_available())
```

## See also

- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.disable.md|pycauset.cuda.disable]]
- [[docs/functions/pycauset.cuda.current_device.md|pycauset.cuda.current_device]]
