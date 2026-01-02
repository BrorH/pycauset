````markdown
# pycauset.set_backing_dir

```python
pycauset.set_backing_dir(path: str | Path) -> Path
```

Sets the directory used for **auto-created backing files** (the temporary on-disk files that hold payload bytes for large matrices/vectors during a session).

## When to call it

Call it **once**, immediately after importing PyCauset, and **before** creating large matrices:

```python
import pycauset as pc

pc.set_backing_dir(r"D:\\pycauset_tmp")

M = pc.zeros((5000, 5000), dtype=pc.int32)
M[0, 0] = 42
```

## Notes on switching

You *can* call this multiple times, but switching the backing directory mid-session is not guaranteed to be stable for already-created objects (some may remain backed by the old directory).


## See also

- [[guides/Storage and Memory.md|Storage and Memory]]
- [[dev/Storage Semantics.md|Storage Semantics]]
````
