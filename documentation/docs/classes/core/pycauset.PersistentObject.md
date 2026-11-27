# Persistent objects (base behavior)

PyCauset objects are typically backed by either RAM or a memory-mapped file, and they can be persisted as `.pycauset` snapshots.

There is no stable public Python class named `pycauset.PersistentObject` in Release 1.

Instead, these behaviors surface through:

- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
- [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]

## Storage concepts

- **Backed by disk or RAM:** the API is the same either way.
- **Snapshots are immutable by default:** `load()` does not implicitly overwrite the file you loaded.
- **Metadata-first:** shape, dtype, view-state and properties are stored without scanning the payload.

The canonical Release 1 persistence semantics and container format are documented in:

- [[guides/Storage and Memory.md|Storage and Memory]]

## Semantic properties

`obj.properties` is a dict of user notes (string keys: a label, a source, a seed), saved with
the object and restored on load. PyCauset never reads it; structure is the class
(`TriangularMatrix`, `SymmetricMatrix`, ...).

## See also

- [[docs/functions/pycauset.save.md|pycauset.save]]
- [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Storage and Memory.md|Storage and Memory]]
