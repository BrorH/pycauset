# Block Matrices

Block matrices provide a **storage-first** way to represent a large matrix as a 2D grid of smaller matrices (“blocks”), potentially with heterogeneous dtypes.

The core goals are:

- Preserve structure and avoid global densification (“no silent densify”).
- Keep compute routed through the existing leaf compute boundary (AutoSolver / device routing).
- Support semi-lazy orchestration (thunked per-block results, evaluated only on triggers).
- Persist block matrices as a snapshot without writing a single giant dense payload.

## Where it lives

Python implementation:

- `python/pycauset/_internal/blockmatrix.py`: `BlockMatrix` container + orchestration (`block_matmul`, `block_add`, `block_sub`, `block_mul`, `block_div`).
- `python/pycauset/_internal/submatrix_view.py`: `SubmatrixView` (no-copy rectangular view).
- `python/pycauset/_internal/thunks.py`: `ThunkBlock` (lazy, cached per-block evaluation).
- `python/pycauset/_internal/persistence.py`: `matrix_type=BLOCK` save/load support via a sidecar directory.

Integration points:

- `python/pycauset/__init__.py`: `pycauset.matrix(...)` block-grid construction disambiguation.
- `python/pycauset/_internal/ops.py`: `pycauset.matmul(a, b)` routes to block orchestration if either operand is a `BlockMatrix`.

## Data model

### `BlockMatrix`

A `BlockMatrix` is a structural container of blocks laid out in a rectangular grid.

Invariants enforced at construction:

- Grid must be rectangular (every block-row has the same number of block-cols).
- All blocks in a **block-row** share the same height.
- All blocks in a **block-col** share the same width.

The container exposes:

- Elementwise indexing via `M[i, j]` / `M.get(i, j)`.
- Block access via `get_block(r, c)` and `set_block(r, c, block)`.
- Partition metadata via `row_partitions` / `col_partitions`.

### `SubmatrixView`

`SubmatrixView(source, row0, col0, rows, cols)` is a lightweight, no-copy rectangle.

- Element reads delegate to the source.
- `repr/str` are structure-only.
- A view-of-a-view composes deterministically into a single view.

In block orchestration, `SubmatrixView` is used to tile operands when block boundaries do not align.

### `ThunkBlock`

A `ThunkBlock` represents a deferred computation that produces a concrete matrix-like object.

- It caches the computed result.
- It is thread-safe for single-eval concurrency.
- It is triggered by element access (`get` / `__getitem__`) or explicit `materialize()`.

Staleness:

- `ThunkBlock` can pin `version` attributes on captured source objects.
- `BlockMatrix` increments its own `version` on `set_block`.
- Not all native matrix types currently expose an automatic mutation version; if a leaf matrix is mutated in-place and does not update a pinned `version`, a previously created thunk may not detect the change.

## Orchestration semantics

### “Once block, always block”

If either operand is a `BlockMatrix`, operations preserve block-ness by returning a `BlockMatrix` result (typically thunked):

- Matmul: `A @ B` or `pycauset.matmul(A, B)`
- Elementwise: `+`, `-`, `*`, `/`

Mixed operands are handled by wrapping the non-block operand as a `1×1` `BlockMatrix`, then refining partitions to align.

### Partition refinement

- `block_matmul` refines the shared dimension using `sorted(set(A.col_partitions) | set(B.row_partitions))`.
- Elementwise ops refine both axes using the union of row/col partitions.

The refinement step creates `SubmatrixView` tiles when necessary.

### Leaf compute boundary

When orchestration reaches “leaf × leaf” matmul between native matrices, it routes through the public dispatch boundary (`pycauset.matmul`) so property-aware conversions (diagonal/triangular) still apply.

## IO accelerator integration

Orchestrated evaluation performs best-effort IO hints:

- Prefetch before using a backing file: `obj.get_accelerator().prefetch(0, size)`
- Discard after a temporary is no longer needed: `discard(0, size)`

These hooks are intentionally best-effort and should never be required for correctness.

## Persistence format

Saving a block matrix uses a **single `.pycauset` container file** plus a sibling **sidecar directory**:

- Container path: `bm.pycauset`
- Sidecar directory: `bm.pycauset.blocks/`

The container stores:

- `matrix_type = "BLOCK"`
- `data_type = "MIXED"`
- `block_manifest` with:
  - `row_partitions`, `col_partitions`
  - `children`: a grid of `{path, payload_uuid}` entries

Child blocks are stored as `block_r{r}_c{c}.pycauset` files in the sidecar directory.

Snapshot integrity:

- Each manifest entry pins the child `payload_uuid`.
- Load validates the pinned UUID; mismatch errors deterministically.

View blocks on save:

- Persisting `SubmatrixView` blocks materializes the view **block-locally** into a small NumPy array, then converts via `native.asarray`.
- This avoids global densification while still producing stable on-disk storage.

## Debugging and traceability

Kernel trace:

- `pycauset._debug_clear_kernel_trace()`
- `pycauset._debug_last_kernel_trace()`

IO trace (separate channel):

- `pycauset._debug_clear_io_trace()`
- `pycauset._debug_last_io_trace()`

For device routing expectations and thunk trigger testing, prefer trace-based integration tests over timing.

## See also

- [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/functions/pycauset.save.md|pycauset.save]] / [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Matrix Guide|Matrix Guide]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[internals/MemoryArchitecture|Memory Architecture]]
