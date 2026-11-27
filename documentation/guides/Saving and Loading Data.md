# Saving and Loading Data

PyCauset puts everything into a single `.pycauset` file: the payload, its dtype and
shape, and any metadata. This page covers saving, loading, and converting between
formats.

## Save a matrix or vector

`pc.save` writes one file; `pc.load` memory-maps it back:

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ], dtype="float64")
v = pc.vector([1, 2, 3])

pc.save(A, "A.pycauset")
pc.save(v, "v.pycauset")

A2 = pc.load("A.pycauset")
v2 = pc.load("v.pycauset")
```

`pc.load` detects the type from the file header, so you get back the same object kind
you saved.

## Convert between formats

`pc.convert_file` moves data between `.pycauset` and NumPy's `.npy`/`.npz`:

```python
# Snapshot -> npy -> snapshot round-trip
pc.convert_file("A.pycauset", "A.npy")
pc.convert_file("A.npy", "A_roundtrip.pycauset")

# Pick a specific array inside an npz
import numpy as np
np.savez("bundle.npz", matrix0=np.eye(3), vector0=np.arange(3.0))
pc.convert_file("bundle.npz", "vec.pycauset", npz_key="vector0")
```

Supported formats are `.pycauset`, `.npy`, and `.npz`, in any direction. `npz_key`
selects a named array inside an archive (defaults to the first key).

## Where files go

Disk-backed objects use a backing directory. The default is a `.pycauset` folder in
your working directory; point it somewhere else once, right after import:

```python
from pathlib import Path
import pycauset as pc

pc.set_backing_dir(Path.cwd() / "pycauset_storage")
```

Two kinds of files live there:

- `.tmp` — session backing files that hold working payloads. They are deleted on
  exit unless you set `pc.keep_temp_files = True` (useful for debugging).
- `.pycauset` — snapshots you write explicitly with `save()`.

## The materialization guard

Turning a large matrix into a dense NumPy array forces all of it into RAM, which can crash
a process. PyCauset guards against this by size: `np.asarray(obj)` raises when the array would
not fit in free RAM, whether the matrix lives in RAM or in a file. Opt out explicitly:

```python
arr = pc.to_numpy(A2, allow_huge=True)    # "yes, I want it in RAM"
```

Exports are guarded by size only: one that would not fit in free RAM raises, as does one above
the ceiling set with `pc.set_export_max_bytes(n)` (`None`: back to the free-RAM rule).
`allow_huge=True` skips the check.

See [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]] and
[[guides/Storage and Memory|Storage and Memory]] for the details.
