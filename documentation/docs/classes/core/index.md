# Core

This section describes behaviors that apply to PyCauset's storage-backed objects.

User-facing base types:

- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
- [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]

Streaming access hints:

- [[docs/classes/core/pycauset.AccessPattern.md|pycauset.AccessPattern]]
- [[docs/classes/core/pycauset.MemoryHint.md|pycauset.MemoryHint]]

Errors:

- [[docs/classes/core/pycauset.StaleThunkError.md|pycauset.StaleThunkError]]: a lazy block
  result read after an input was written

For persistence and snapshots, see:

- [[guides/Storage and Memory.md|Storage and Memory]]

