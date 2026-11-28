# Block Matrices

Block matrices provide a **storage-first** way to represent a large matrix as a 2D grid of smaller matrices (“blocks”), potentially with heterogeneous dtypes.

The core goals are:

- Preserve structure and avoid global densification (“no silent densify”).
- Keep compute routed through the ordinary operations on the leaves (one engine path per operation).
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

- Elementwise indexing via `M[i, j]`.
- Block access via `get_block(r, c)` and `set_block(r, c, block)`.
- Partition metadata via `row_partitions` / `col_partitions`.

### `SubmatrixView`

`SubmatrixView(source, row0, col0, rows, cols)` is a lightweight, no-copy rectangle.

- Element reads delegate to the source.
- `repr/str` are structure-only.
- A view-of-a-view composes deterministically into a single view.

In block orchestration, `SubmatrixView` is used to tile operands when block boundaries do not
align. When a tile is computed, it is the engine's slice of its leaf: a zero-copy window of a
dense leaf, a dense copy of just the tile of a structured one, a sub-grid of a nested block
matrix. A tile of a duck-typed matrix-like leaf raises `TypeError`: leaves that operations cut
must be PyCauset matrices.

Slicing a block matrix, `bm[r0:r1, c0:c1]` (step 1), refines at the slice bounds and returns
the sub-grid of those tiles (no copy). Other slices raise `TypeError`.

### `ThunkBlock`

A `ThunkBlock` represents a deferred computation that produces a concrete matrix-like object.

- It caches the computed result.
- It is thread-safe for single-eval concurrency.
- It is triggered by element access (`get` / `__getitem__`) or explicit `materialize()`.

Staleness (snapshot-at-creation, R1):

- `ThunkBlock` pins the versions of everything it reads at creation (its leaves, the leaves of
  nested grids and cut tiles, a thunk operand's own pins); evaluation and cache hits check them
  and **raise** `pycauset.StaleThunkError` on a mismatch ("recompute the operation"; no
  auto-recompute). Its cache is the one result cache outside the derived-value cache
  (review/features/DERIVED_CACHE.md).
- `BlockMatrix` increments its own `version` on `set_block`, invalidating cached/thunked blocks owned by the container.
- Engine leaves are pinned by their storage version, so writing a leaf (`A[0, 0] = 1`) makes
  every thunk that read it stale; nothing has to be bumped by hand. A duck-typed leaf takes part
  through its own `version` attribute, if it has one.

## Orchestration semantics

### “Once block, always block”

If either operand is a `BlockMatrix`, operations preserve block-ness by returning a `BlockMatrix` result (typically thunked):

- Matmul: `A @ B` or `pycauset.matmul(A, B)`
- Elementwise: `+`, `-`, `*`, `/`, with another matrix or a scalar (a scalar maps over the
  leaves, in the grid's dtype), and `-A`
- Transposes: `A.T`, `A.conj()`, `A.H` (the grid transposed, each leaf an engine view)

A matrix operand that is not a block matrix is wrapped as a `1×1` `BlockMatrix`, then the
partitions are refined to align. Vectors are not block operands (`TypeError`; use an n × 1
matrix).

### Partition refinement

- `block_matmul` refines the shared dimension using `sorted(set(A.col_partitions) | set(B.row_partitions))`.
- Elementwise ops refine both axes using the union of row/col partitions.

The refinement step creates `SubmatrixView` tiles when necessary.

### Leaf compute boundary

A "leaf × leaf" matmul of two engine operands (matrices, cut tiles) goes through
`pycauset.matmul`, so the engine's structure rules and its out-of-core decision apply to each
leaf product. Other operands (nested block matrices, other matrix-likes) use their own
`@`. The choice is by type (`_leaf_matmul`), never by trying one path and catching its error.
Each leaf product is an ordinary engine operation
([[internals/Compute Architecture.md|Compute Architecture]]).

### One dtype per grid

A block matrix has one dtype, `bm.dtype`: the engine's promotion over its leaves (as for the
operands of an addition; DType System). Every block operation resolves its result dtype once
from the two grids' dtypes, before evaluation, and each tile casts its leaves to it in the
engine first, so a block result equals the same operation on the assembled matrices (an int8
tile beside a float32 one computes in float32). The export (`np.asarray`, `to_numpy`) has the
grid's dtype. A grid with a matrix-like leaf that has no PyCauset dtype is `"mixed"`, and its
tiles keep their own operations.

### Evaluation triggers (semi-lazy)

- Trigger evaluation of the minimal required block(s): element access, crossing the compute boundary, dense conversion (`np.asarray`), or persistence (`pycauset.save`).
- Non-triggers: `repr/str`, shape/partition metadata, and `get_block`.
- Cached results are reused until a version mismatch is detected; stale hits raise.

Concurrency: each `ThunkBlock` evaluates once under a `threading.Lock`, so concurrent requests
compute once and reuse the cached block.

### Deterministic accumulation per output block

- Fixed `k` order for `Σ_k A_ik @ B_kj`.
- The result dtype is the engine's matmul dtype of the two grids' dtypes (see above); every
  term computes in it, so the sum is in it too.

## IO accelerator integration

Orchestrated evaluation performs best-effort IO hints:

- Prefetch the leaves before a product: `obj._io_prefetch()`
- Discard only temporaries: the previous accumulator, and a product that is not the
  accumulator. Inputs and fresh results stay resident (the caller reads them next).

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

Save policies (Release 1):

- Stale thunks fail save deterministically (no implicit recompute).
- Overwrite cleanup deletes only deterministic child filenames within the sidecar.
- Saves stage child files (and nested sidecars) then commit/rename to reduce partial updates.

View blocks on save:

- Persisting a `SubmatrixView` block of a dense leaf writes it from an engine window; other
  view blocks are copied block-locally (no global densification).
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
- [[internals/Memory and Data|Memory & Data]]
