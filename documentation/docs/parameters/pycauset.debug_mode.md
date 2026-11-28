# pycauset.debug_mode

```python
pycauset.debug_mode = True
```

When `True`, PyCauset prints one diagnostic line per event to stderr, prefixed with
`[pycauset]`: `matmul`, `invert`, `solve`, `save` and `load` with their operands (shape,
dtype, RAM or file) and wall time, and changes to the memory threshold and
backing directory. Off by default.

For a record you can inspect from code, use [[docs/functions/pycauset.last_io_trace.md|pycauset.last_io_trace]].

## Properties

*   **Type**: *bool*
*   **Default**: `False`
