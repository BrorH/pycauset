
# pycauset.keep_temp_files

```python
pycauset.keep_temp_files
```

Controls whether PyCauset deletes its temporary disk-backed storage files when the Python process exits.

When `False` (default), temporary files created during a session are cleaned up automatically.

Set to `True` when debugging persistence issues, inspecting intermediate `.pycauset` artifacts, or reproducing a disk-backed performance scenario.

## Properties

*   **Type**: *bool*
*   **Default**: `False`

## Notes

- This is a global setting that affects the runtime cleanup behavior.
- It does not automatically “save” objects as portable archives; use `pycauset.save(...)` for that.
