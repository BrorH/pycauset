# R1 Storage (Persistence Container)

Release 1 ships a single-file `.pycauset` persistence container that is:

- **mmap-friendly** for large payloads,
- **crash-consistent** for metadata updates,
- **forward-compatible** via sparse, typed metadata,
- and designed to support scale-first workflows (out-of-core objects without full materialization).

This guide focuses on the user-facing semantics: save/load, snapshot immutability, and what metadata is preserved.

## Minimal example

```python
import pycauset as pc

A = pc.zeros((128, 64), dtype="float32")
A.fill(1.0)

pc.save(A, "A.pycauset")
B = pc.load("A.pycauset")

assert B.shape == (128, 64)
```

## Snapshots and mutation (Release 1 semantics)

In Release 1:

- A `.pycauset` file is treated as an **immutable snapshot**.
- `pycauset.load(path)` returns a snapshot-backed object.
- Mutating the loaded object does **not** implicitly write back to `path`.
- To persist changes, explicitly save a new snapshot.

This protects expensive “baseline” artifacts from accidental overwrite.

The canonical description (including cached-derived values and big-blob caches) is in:

- [[guides/Storage and Memory.md|Storage and Memory]]

## What is persisted

Release 1 persistence round-trips:

- identity metadata (shape, dtype, matrix type, payload layout)
- view-state metadata (transpose/conjugation/scalar)
- user-facing semantic properties (`properties.*`, gospel)
- cached-derived values (`cached.*`, validity-checked)

This means:

- NxM shapes are preserved (`rows`, `cols`).
- Transpose is preserved as metadata (no forced densification).
- Properties and caches are restored when valid.

### Block matrices

Block matrices persist as a manifest plus child files (sidecar directory `path + ".blocks"`):

- `matrix_type="BLOCK"`, `data_type="MIXED"` in the container header.
- Manifest pins `row_partitions` / `col_partitions` and a grid of child references (`path`, `payload_uuid`).
- Child filenames are deterministic (`block_r{r}_c{c}.pycauset`) and written under the sidecar directory; overwrite cleanup deletes only matching child names.
- Saves evaluate thunk blocks **blockwise** (never global densify), materialize `SubmatrixView` blocks locally, and raise deterministically on stale thunks.
- Saves stage child files (and nested sidecars) before commit; `payload_uuid` pins make mixed-snapshot loads fail deterministically.
- There is no block-level `trace/determinant/norm/sum` cache in Release 1; cached-derived values remain per-leaf child only.

## Failure modes and constraints

- Version mismatches or corrupted headers/metadata fail deterministically (clear error).
- Payload is not scanned during load to “validate” metadata.
- Missing/corrupt big-blob cache objects are treated as cache misses (may emit a storage warning; base object still loads).

## Where the on-disk format is specified

The authoritative `.pycauset` container format specification is documented in:

- [[dev/PyCauset Container Format.md|PyCauset Container Format]]

For user-facing semantics (save/load workflows, snapshot behavior, caches), see:

- [[guides/Storage and Memory.md|Storage and Memory]]

For contributor-level details and debugging tools, see:

- [[dev/Storage Semantics.md|Storage Semantics]]

## See also

- [[docs/functions/pycauset.save.md|pycauset.save]]
- [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[guides/release1/properties.md|R1 Properties]]
