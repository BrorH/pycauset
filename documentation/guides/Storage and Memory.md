# Storage and Memory

PyCauset lets you work with matrices that don’t fit in RAM by storing their data on disk and letting the operating system load pieces as you touch them.

If you’re comfortable with NumPy, this will feel familiar: you use normal indexing (`M[i, j]`), shapes, and transpose views (`M.T`). Nothing “auto-saves” behind your back, saving is explicit.

Most of the time you don’t need to think about file layouts or memory maps. You just create matrices, use normal indexing (`M[i, j]`), and call `save()` / `load()`.

In examples below, assume `import pycauset as pc` unless shown.

## Quickstart: save and load

### Create → save → load

```python
import pycauset as pc

M = pc.zeros((5000, 5000), dtype=pc.int32)
M[0, 0] = 42

pc.save(M, "my_matrix.pycauset")
N = pc.load("my_matrix.pycauset")

assert N[0, 0] == 42
```

## A simple mental model: “data bytes” + “info bytes”

Most PyCauset objects have the same basic structure on disk, and the stored file can be broken into two sections:

- **Payload (data bytes):** the matrix/vector entries themselves.
- **Metadata (info bytes):** small structured information that explains what the payload means (shape, dtype, layout) and how to interpret it (transpose, scalar, conjugation).

The important consequence is that PyCauset can scale because it does *not* require the payload to be in RAM all at once. The operating system will automatically keep recently-used parts in memory.

### A concrete example

Suppose you create and save a matrix:

```python
M = pc.zeros((2, 3), dtype="float32")
M[0, 1] = 20
pc.save(M, "M.pycauset")
```

Conceptually, the saved file contains:

- **Payload (data bytes):** the raw 2×3 float32 numbers stored as bits.
- **Metadata (info bytes):** small information like:
   - shape: 2 rows, 3 cols
   - dtype: float32
   - view-state: “not transposed”, “scalar = 1.0”, etc.

You can think of metadata like the “label on the box” and payload like “what’s inside the box”.

### Why this matters

This specific format enables two user-facing behaviors:

1) **Gigantic matrices actually work:** the payload lives on disk and is paged in as needed.
2) **Some operations are instant:** things like transpose or scaling can be represented by changing metadata (no rewrite of the large payload).

## Structure and metadata

What a saved object carries: shape, dtype, its structure (the class: `TriangularMatrix`,
`SymmetricMatrix`, `HermitianMatrix`, `DiagonalMatrix`, ...; or build one with
`pc.symmetric(...)`, `pc.hermitian(...)`, `pc.diagonal(...)`), the view state, and for causal
sets the spacetime and seed. Structure is part of the storage, so it cannot disagree with the
values.

Computed results are not cached yet; a result always reflects the current values, also after
`M[i, j] = v`. A cache that is discarded on every change is planned (review phase D).

## Where backing files go (temporary storage)

A **backing file** is the on-disk file that holds a matrix’s payload bytes while you work.

Why it’s needed:

- It lets PyCauset memory-map the payload, so matrices can be larger than RAM.
- It gives the OS something concrete to page in/out (instead of forcing PyCauset to keep all bytes in Python-managed memory).

When PyCauset needs to create these backing files automatically, it uses a storage root directory:

- Default: a `.pycauset/` directory under your current working directory.
- To change it: call `pc.set_backing_dir(...)` once after import (and ideally before allocating large matrices).

Example:

```python
import pycauset as pc
# Call once, early in your program.
pc.set_backing_dir(r"D:\\pycauset_tmp")
```

PyCauset cleans up temporary backing files (extensions like `.tmp` / `.raw_tmp`) in two places:

- **On import** (startup): removes potential leftovers from previous runs.
- **On interpreter exit**: removes temporary files from the current run.

Note: `pc.keep_temp_files = True` prevents the exit-time cleanup. Startup cleanup still runs so you don't accidentally reuse stale temp files.

If you want to keep them around for debugging:

```python
pc.keep_temp_files = True # Files will not be deleted on program exit
```

## Spill (“switch to file-backed mapping”)

When a matrix starts out in RAM, PyCauset may later **spill** it to disk to free RAM.

In concrete terms, **spill** means that the live object switches from a *RAM-only* mapper to a **file-backed (memory-mapped) mapper**.

Important details:

- Spilling writes the **payload bytes only** to a temporary `.tmp` backing file under the current backing dir.
- The live Python object keeps its **metadata** in memory.
- Spilling does **not** create a `.pycauset` snapshot container. If you want the object to remain on your disk, you must explicitly call `save()`.

## `.tmp` vs `.raw_tmp` vs `.pycauset`

You may see these extensions in your backing directory:

- `.tmp`: temporary backing files that hold **payload bytes** during a session (including spill/eviction and large auto-allocations).
- `.raw_tmp`: a staging file used while writing a snapshot via `save()`. It is only renamed/committed to `.pycauset` when the write completes.
- `.pycauset`: a **snapshot container** you created explicitly via `save()`.

## NumPy conversion safety: when materialization is allowed

PyCauset protects you from accidental full materialization when converting to NumPy:

- **Snapshot-backed objects** (`.pycauset`): `np.asarray(obj)` is allowed and returns a copy.
- **RAM-backed objects** (`:memory:`): `np.asarray(obj)` is allowed and stays in-memory.
- **Spill/file-backed objects** (e.g., `.tmp`): `np.asarray(obj)` **raises** by default to avoid surprise RAM blow-ups. Opt in explicitly via `pc.to_numpy(obj, allow_huge=True)`.
- **Ceiling control**: `pc.set_export_max_bytes(bytes_or_None)` sets a materialization limit for NumPy exports; `None` disables the size ceiling (file-backed safety still applies unless you pass `allow_huge=True`).

Takeaway: if something is spill-backed and large, you must opt-in to materialize; snapshots and in-RAM objects remain convertible without the opt-in.

## Out-of-core execution and IO traces

Operations whose operands and result do not fit the RAM budget run tile by tile through the
engine's StreamingManager; others run directly on memory-mapped storage. The engine decides
(see [[internals/Streaming Manager|Streaming Manager]]); there is no routing knob.

`pc.last_io_trace(op)` reports what a call read and how it ran: each operand's shape, dtype
and storage (`"memory"`, `"file"`, `"temp"`), the kernels the engine ran, and `streamed`
([[docs/functions/pycauset.last_io_trace.md|pycauset.last_io_trace]]).

## Snapshots vs working copies

The biggest user-facing rule is: a `.pycauset` file is treated as an **immutable snapshot**.

A "snapshot" is a saved, read-only point-in-time artifact. Loading a snapshot gives you an object that *reads from* that file, but editing the object does not “edit the file”.

This is intentionally NumPy-like: `np.load(...)` gives you an array in memory; mutating it doesn’t rewrite the file on disk. PyCauset keeps the same principle, even though the payload may be disk-backed.

- `pc.load(path)` gives you an object backed by that snapshot.
- Mutating the object does **not** implicitly overwrite the object saved at `path`.
- If you want a new persisted version, explicitly `pc.save(obj, new_path)`.

Why this exists: it protects expensive “baseline” snapshots from accidental overwrite during exploratory work.

## Metadata-driven “views” (fast operations without rewriting payload)

A **view** is a matrix/vector object that **shares the same payload bytes** as another object, but carries different *metadata* that changes how those bytes are interpreted.

Creating a view is often $O(1)$ because PyCauset only needs to allocate a small wrapper + update metadata. The large payload file is not rewritten or copied.
This is the same basic idea as NumPy’s transpose: `M.T` is not a deep copy.

### What counts as a view?

These are typical metadata-only transforms:

- **Transpose**: toggles a transpose flag (e.g. `M.T` / `M.transpose()`).
- **Conjugation**: toggles a conjugation flag (e.g. `M.conj()`).
- **Scalar scaling**: stores a scalar multiplier in metadata (e.g. `3 * M`).

### Where does the cost go?

The “work” is deferred until you actually read/compute:

- On element access `V[i, j]`, PyCauset maps that request back to the base payload and applies the view-state.

Concrete intuition:

```python
M = pc.zeros((2, 3), dtype="float32")
V = M.T

# Same payload; different interpretation:
# V[i, j] reads the bytes for M[j, i].
assert V[1, 0] == M[0, 1]
```

This is why the container format reserves a `view` namespace in metadata: view-state is part of the object’s meaning, and it must survive `save()` / `load()`.

## Multi-file snapshots: BlockMatrix sidecars

Most objects persist as a single `.pycauset` file. Block matrices are the main exception.

When you save a block matrix, PyCauset writes:

- the container file, and
- a sibling sidecar directory `path + ".blocks"` containing child block snapshots.

The block manifest pins each child’s `payload_uuid` so mixed-snapshot loads fail deterministically.

Practical rule:

- To move/copy a saved block matrix, copy `X.pycauset` **and** `X.pycauset.blocks/`.

## Format interoperability (what we support)

PyCauset ships one canonical snapshot format and a minimal NumPy bridge for pipelines:

- `.pycauset`: canonical snapshot container (always supported).
- `.npy` / `.npz`: import/export supported via `pc.convert_file`, `pc.load_npy`, `pc.load_npz`, `pc.save_npy`, `pc.save_npz`.

Example (convert snapshot → npy → snapshot without materializing into Python first):

```python
pc.save(pc.zeros((3, 3)), "A.pycauset")
pc.convert_file("A.pycauset", "A.npy")
pc.convert_file("A.npy", "A_roundtrip.pycauset")
```

Routing rules and guardrails:

- `.npy`/`.npz` exports honor the same materialization safety as `np.asarray`: file-backed objects require `allow_huge=True` to load into RAM; otherwise export raises.
- `.npz` imports default to the first key; set `npz_key` to pick a named entry.
- `pc.convert_file(src, dst, ...)` infers formats from suffixes and converts between `.pycauset` and `.npy`/`.npz` without you having to write load/save boilerplate.

Formats we are considering (not implemented in R1): MatrixMarket `.mtx`, MATLAB `.mat`, Parquet/Arrow/CSV for tabular interop. These remain future work until documented otherwise.

### Future format targets (not implemented yet)

- **MatrixMarket `.mtx`**: common sparse/text interchange; useful for scientific benchmarks.
- **MATLAB `.mat`**: prevalent in engineering/scientific workflows.
- **Parquet / Arrow / CSV**: tabular interop for pandas-style pipelines; CSV primarily for debugging/sanity checks.
- **HDF5/NetCDF** (under evaluation): only if a stable, low-maintenance reader fits the maintenance budget.

These are roadmap candidates; support will be added only when implemented and documented.

## File format and debugging (advanced)

If you’re writing tooling, debugging a corrupted file, or just curious about the exact on-disk layout, see:

- [[dev/PyCauset Container Format.md|PyCauset Container Format]] (canonical `.pycauset` format spec)
- [[dev/Storage Semantics.md|Storage Semantics]] (developer-facing semantics and runbooks)

### Copying snapshots correctly

- For a plain matrix snapshot, copying the single `X.pycauset` file is sufficient.
- For block matrices, also copy `X.pycauset.blocks/`.

## Memory efficiency (why causal sets fit)

Storage format is only half the story: PyCauset also uses domain-specific representations (bit-packing, triangular storage) so the payload itself is smaller.

See:

- [[guides/Causal Sets.md|Causal Sets]]
- [[guides/Matrix Guide.md|Matrix Guide]]
- [[guides/Performance Guide.md|Performance Guide]]

## NumPy Interop & Memory Risks

While PyCauset matrices can be terabytes in size, standard NumPy arrays must fit largely in RAM.

**Interaction Warning:** passing a huge file-backed matrix to `np.array(M)` or `pycauset.to_numpy(M)` attempts to load the **entire payload** into system RAM.

To prevent accidental crashes:
1.  PyCauset blocks this by default for disk-backed objects.
2.  You must explicitly opt-in with `pycauset.to_numpy(M, allow_huge=True)`.
3.  Prefer processing data within PyCauset where possible, as it manages tiling and paging automatically.

See [[guides/Numpy Integration.md|Numpy Integration]] for details.

## See also

- [[docs/functions/pycauset.save.md|pycauset.save]]
- [[docs/functions/pycauset.load.md|pycauset.load]]
- [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]]
- [[dev/Storage Semantics.md|Storage Semantics]]
- [[internals/Memory and Data.md|Memory and Data]]
- [[dev/PyCauset Container Format|PyCauset Container Format]]
- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]

## Crash Consistency (R1_SAFETY)

PyCauset employs several mechanisms to minimize data loss in the event of a power failure or crash:

1.  **Atomic Metadata Updates**: .pycauset files use double-buffered metadata slots. Updates are written to a new slot, flushed to disk, and then the "Active Slot" pointer is updated atomically.
2.  **Explicit Flushes**: Critical write operations (like save() or internal spills) call FlushFileBuffers (Windows) or msync (Linux) to ensure data reaches physical media.
3.  **Header Protection**: All backing files include a versioned header to prevent partial or corrupt files from being misinterpreted as valid data.
