# Saving and Loading Data

PyCauset puts everything into a single `.pycauset` file: the payload, its dtype and
shape, and any metadata. This page covers saving, loading, and converting between
formats.

## Save a causal set

`CausalSet` has a `save` method, and `pc.load` reads it back:

```python
import pycauset as pc

c = pc.causet(n=1000, seed=42)
c.save("universe.pycauset")

again = pc.load("universe.pycauset")
print(again.n)          # 1000
```

The file holds the causal matrix, the coordinates, and the metadata (spacetime, seed,
parameters), so `pc.load` rebuilds the object without re-sprinkling.

## Save a raw matrix or vector

The same functions work for matrices and vectors:

```python
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

Turning a large disk-backed object into a dense NumPy array forces it into RAM, which
can crash a process. PyCauset guards against this:

- Snapshot-backed (`.pycauset`) and RAM-backed objects convert freely with
  `np.asarray(obj)`.
- Spill/file-backed (`.tmp`) objects raise unless you opt in explicitly:

```python
arr = pc.to_numpy(obj, allow_huge=True)   # "yes, I want it in RAM"
```

`pc.set_export_max_bytes(n)` sets a ceiling on materialization; `None` removes the
limit (file-backed objects still need `allow_huge=True`).

See [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]] and
[[guides/Storage and Memory|Storage and Memory]] for the details.
