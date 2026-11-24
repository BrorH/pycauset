# pycauset.cuda.current_device

```python
pycauset.cuda.current_device() -> str
```

Returns the name of the current compute device.

If CUDA is inactive, this returns a CPU-only identifier.

## Returns

*   **str**: Device name.

## Notes

*   The string is a high-level label (e.g., `AutoSolver (CPU Only)` or
	`AutoSolver (CPU + CUDA (NVIDIA GPU))`).
*   For detailed hardware properties, call `pycauset.cuda.benchmark()`.

## Exceptions / warnings

*   Returns a CPU identifier if CUDA is inactive.

## Example

```python
import pycauset as pc

print(pc.cuda.current_device())

if pc.cuda.is_available():
    print("GPU active")
else:
    print("CPU-only routing")
```

## See also

- [[docs/functions/pycauset.cuda.is_available.md|pycauset.cuda.is_available]]
- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.disable.md|pycauset.cuda.disable]]
