# pycauset.is_gpu_available

```python
pycauset.is_gpu_available() -> bool
```

Return `True` when the CUDA backend is loaded and a GPU device is active.

This is the one-line availability check; the fuller control surface lives on
[[pycauset.cuda.is_available|pycauset.cuda.is_available]]. On a CPU-only install
this returns `False` and all operations route to the CPU backend.

## Returns

*   **bool**: `True` if a GPU device is active, `False` otherwise.
