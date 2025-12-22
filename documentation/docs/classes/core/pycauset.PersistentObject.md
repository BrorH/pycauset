# Persistent objects (base behavior)

PyCauset objects are typically backed by either RAM or a memory-mapped file, and they can be persisted as `.pycauset` snapshots.

There is no stable public Python class named `pycauset.PersistentObject` in Release 1.

Instead, these behaviors surface through:

- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
- [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]

## Storage concepts

- **Backed by disk or RAM:** the API is the same either way.
- **Snapshots are immutable by default:** `load()` does not implicitly overwrite the file you loaded.
- **Metadata-first:** shape, dtype, view-state, and semantic properties are stored/propagated without scanning payload.

The canonical Release 1 persistence semantics and container format are documented in:

- [[guides/Storage and Memory.md|Storage and Memory]]

## Semantic properties

Matrices and vectors expose `obj.properties`, a typed mapping used for:

- gospel semantic assertions (e.g. `is_upper_triangular=True`), and
- cached-derived values (e.g. `trace`, `determinant`, `norm`) with strict validity.

See [[guides/release1/properties.md|R1 Properties]] for the user-facing contract.

## See also

- [[docs/functions/pycauset.save.md|pycauset.save]]
- [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[guides/release1/properties.md|R1 Properties]]
