# Storage and Memory

PyCauset lets you work with matrices that don’t fit in RAM by storing their data on disk and letting the operating system load pieces as you touch them.

If you’re comfortable with NumPy, this will feel familiar: you use normal indexing (`M[i, j]`), shapes, and transpose views (`M.T`). Nothing “auto-saves” behind your back — saving is explicit.

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
- **Metadata (info bytes):** small structured information that explains what the payload means (shape, dtype, layout) and how to interpret it (transpose, scalar, conjugation), plus optional properties and caches.

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
   - optional: user properties and cached results

You can think of metadata like the “label on the box” and payload like “what’s inside the box”.

### Why this matters

This specific format enables two user-facing behaviors:

1) **Gigantic matrices actually work:** the payload lives on disk and is paged in as needed.
2) **Some operations are instant:** things like transpose or scaling can be represented by changing metadata (no rewrite of the large payload).

## Semantic Properties (Metadata)

In addition to physical metadata (shape/dtype), Release 1 allows attaching **Semantic Properties**. These are asserted facts about the matrix that are "gospel" (authoritative) and can drastically speed up computations.

### What `properties` is

- `obj.properties` is a mapping (`str` keys → typed values) exposed on every matrix/vector.
- It stores **gospel assertions** (e.g., `is_upper_triangular=True`) which the system accepts without truth-validation.
- It also stores **cached-derived values** (e.g., `determinant`) which are invalidated on mutation.

### Common keys (Release 1)

These keys are treated as **semantic structure claims** (no truth validation):
- **Structure:** `is_symmetric`, `is_hermitian`, `is_upper_triangular`, `is_lower_triangular`, `is_diagonal`
- **Identity:** `is_identity` (implies diagonal, symmetric, etc.)
- **Vector hints:** `is_sorted`, `is_unit_norm`

Example:
```python
A = pc.identity(3)
# Gospel assertion: solver is allowed to treat off-triangle entries as zero.
A.properties["is_upper_triangular"] = True
```

### Technical Implementation (Python-Managed)

Currently, the property system is implemented primarily in the **Python layer**:
- The `properties` dictionary is injected into the native object at runtime (via `python/pycauset/_internal/properties.py`).
- C++ kernels currently see physical metadata (shape/dtype) but not these semantic properties directly.
- **Future Integration:** A mechanism to mirror these high-impact flags to C++ (e.g., via a bitmask) is planned so that backend drivers (like the GPU engine) can inspect them without crossing the Python boundary.

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
- The live Python object keeps its **metadata, properties, and any already-computed cached values** in memory.
- Spilling does **not** create a `.pycauset` snapshot container. If you want the object to remain on your disk, you must explicitly call `save()`.

## `.tmp` vs `.raw_tmp` vs `.pycauset`

You may see these extensions in your backing directory:

- `.tmp`: temporary backing files that hold **payload bytes** during a session (including spill/eviction and large auto-allocations).
- `.raw_tmp`: a staging file used while writing a snapshot via `save()`. It is only renamed/committed to `.pycauset` when the write completes.
- `.pycauset`: a **snapshot container** you created explicitly via `save()` (plus optional sibling cached objects like `X.pycauset.objects/...`).

## NumPy conversion safety: when materialization is allowed

PyCauset protects you from accidental full materialization when converting to NumPy:

- **Snapshot-backed objects** (`.pycauset`): `np.asarray(obj)` is allowed and returns a copy.
- **RAM-backed objects** (`:memory:`): `np.asarray(obj)` is allowed and stays in-memory.
- **Spill/file-backed objects** (e.g., `.tmp`): `np.asarray(obj)` **raises** by default to avoid surprise RAM blow-ups. Opt in explicitly via `pc.to_numpy(obj, allow_huge=True)`.
- **Ceiling control**: `pc.set_export_max_bytes(bytes_or_None)` sets a materialization limit for NumPy exports; `None` disables the size ceiling (file-backed safety still applies unless you pass `allow_huge=True`).

Takeaway: if something is spill-backed and large, you must opt-in to materialize; snapshots and in-RAM objects remain convertible without the opt-in.

## Streaming manager: routing and observability

PyCauset routes large or file-backed operations through a streaming manager so out-of-core work is predictable and observable.

- **Threshold-based routing:** set the IO streaming threshold via `pc.set_io_streaming_threshold(bytes_or_None)`. File-backed operands always stream; `allow_huge=True` on an op bypasses the threshold check.
- **Per-op descriptors:** `matmul`, `invert`, `eigvalsh`, `eigh`, and `eigvals_arnoldi` publish access patterns and tiling/queue hints. Non-square eig/invert inputs are forced to the direct route via a guard.
- **Plan + events:** the last plan is available through `pc.last_io_trace(...)` and includes `{route, reason, tile_shape, queue_depth, plan.access_pattern, events}`. Prefetch/discard events and `impl=...` markers show which path executed.
- **Default behavior:** when thresholds are tiny, tiles shrink automatically and queue depth stays bounded. When thresholds are disabled (`None`), routes stay direct unless operands are file-backed.

## Snapshots vs working copies

The biggest user-facing rule is: a `.pycauset` file is treated as an **immutable snapshot**.

A "snapshot" is a saved, read-only point-in-time artifact. Loading a snapshot gives you an object that *reads from* that file, but editing the object does not “edit the file”.

This is intentionally NumPy-like: `np.load(...)` gives you an array in memory; mutating it doesn’t rewrite the file on disk. PyCauset keeps the same principle, even though the payload may be disk-backed.

- `pc.load(path)` gives you an object backed by that snapshot.
- Mutating the object does **not** implicitly overwrite the object saved at `path`.
- If you want a new persisted version, explicitly `pc.save(obj, new_path)`.

Why this exists: it protects expensive “baseline” snapshots (and their cached artifacts) from accidental overwrite during exploratory work.

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

## Properties and caches: what gets remembered (and when it is trusted)

PyCauset separates two ideas that often get mixed up:

### 1) Gospel properties (`obj.properties`)

`obj.properties` is a power-user escape hatch for *semantic assertions* that algorithms are allowed to trust.

- Example: if you set `obj.properties["is_upper_triangular"] = True`, structure-aware algorithms may treat entries below the diagonal as zero.
- PyCauset does not scan payload data to validate these assertions.
- Booleans are tri-state via key presence: “unknown” = key absent, “known false” = key present with `False`.

### 2) Cached-derived values (compute-once)

Cached-derived values are results PyCauset can recompute from payload + view-state (trace, determinant, norm, etc.).

They are only used when they’re known to match the current object state. Cache validity is checked in $O(1)$ via a signature:

- `payload_uuid`: identity of the persisted payload bytes.
- `view_signature`: compact signature derived from view-state.

On disk, cached-derived values live under `cached.*` (with their signature). At runtime they’re surfaced via `obj.properties` for convenience.

### Big-blob caches (e.g., persisted inverse)

Some cached results are too large to store directly inside metadata. In that case, PyCauset stores them as separate sibling `.pycauset` objects and links to them from the base snapshot.

Example:

```python
pc.save(pc.FloatMatrix(2), "A.pycauset")
A = pc.load("A.pycauset")

# Persist the inverse as a sibling cached object.
inv = A.invert(save=True)
```

On disk, the cached object lives in a sibling directory:

- Base snapshot: `A.pycauset`
- Object store: `A.pycauset.objects/<object_id>.pycauset`

Failure behavior:

- If a referenced cached object is missing or unreadable, PyCauset emits `PyCausetStorageWarning` and treats that cache entry as unusable.
- PyCauset does **not** implicitly recompute or “repair” missing big-blob cache objects. If you want that cached result again, you must request it explicitly (for example, `A.invert(save=True)` to rebuild a missing inverse cache).

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
- For persisted big-blob caches (inverse, etc.), also copy `X.pycauset.objects/`.

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
- [[guides/release1/storage.md|R1 Storage]]
- [[dev/Storage Semantics.md|Storage Semantics]]
- [[internals/MemoryArchitecture.md|MemoryArchitecture]]
- [[internals/Memory and Data.md|Memory and Data]]
- [[internals/plans/completed/R1_STORAGE_PLAN.md|R1_STORAGE_PLAN]]
- [[internals/plans/completed/R1_PROPERTIES_PLAN.md|R1_PROPERTIES_PLAN]]
- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]

## Crash Consistency (R1_SAFETY)

PyCauset employs several mechanisms to minimize data loss in the event of a power failure or crash:

1.  **Atomic Metadata Updates**: .pycauset files use double-buffered metadata slots. Updates are written to a new slot, flushed to disk, and then the "Active Slot" pointer is updated atomically.
2.  **Explicit Flushes**: Critical write operations (like save() or internal spills) call FlushFileBuffers (Windows) or msync (Linux) to ensure data reaches physical media.
3.  **Header Protection**: All backing files include a versioned header to prevent partial or corrupt files from being misinterpreted as valid data.
