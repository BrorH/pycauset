# R1_BLOCKMATRIX Plan — Block Matrices + Heterogeneous Dtypes (Storage-First)

**Status:** Active (drafted 2025-12-17)

This plan is the canonical source of truth for block matrices in Release 1.

!!! warning "Active plan (not shipped)"
    The behavior in this document is not guaranteed to exist in the current codebase.
    Do not add user-facing Guides or API Reference that implies this feature is available until implementation lands (Phase G).

## Canonical dependencies (read before implementing)

- Roadmap node: [PyCauset Roadmap](../TODO.md)
- Foundations:
  - [Compute Architecture](../../Compute%20Architecture.md)
  - [DType System](../../DType%20System.md)
- Release 1 foundations (implemented behavior elsewhere in the system):
  - [Storage](../../../guides/release1/storage.md)
  - [Properties](../../../guides/release1/properties.md)
  - [DTypes](../../../guides/release1/dtypes.md)
  - [Linear Algebra](../../../guides/release1/linalg.md)

## Progress snapshot (MUST keep updated)

**Last updated:** 2025-12-22

**Current phase step:** Phase H — Testing & hardening (completed 2025-12-22)

**What is done (DONE):**

- Phase A contract lock completed (interfaces, invariants, determinism, persistence shape, and staleness semantics).
- Phase B completed: internal `BlockMatrix` container implemented (Python layer) with construction validation, element reads, and structure-only `repr`/`str`.
- Phase B tests added for construction/indexing/printing validation.
- Phase C started: internal `SubmatrixView` container implemented (Python layer) with view composition, dense + block sources, and structure-only printing.
- Phase C tests added for dense + block view element reads, cross-block views, and repr/str non-trigger behavior.
- Phase C refinement implemented (Python layer): `BlockMatrix.refine_partitions(...)` produces a refined block grid using `SubmatrixView` without densification; deterministic validation errors are enforced.
- Phase C refinement tests added (correctness + error cases).
- Phase D started (Python layer): `ThunkBlock` implemented with structure-only printing, lazy evaluation on element access, caching, and snapshot-at-creation staleness checks based on `version` pins.
- Phase D started (Python layer): internal `block_matmul(A,B)` orchestrator produces a BlockMatrix of thunk blocks and refines inner partitions deterministically.
- Phase D tests added for trigger/non-trigger behavior, deterministic k-order accumulation, and stale-thunk errors.
- Phase D elementwise started (Python layer): internal `block_add(A,B)` aligns partitions by refinement-union and returns a BlockMatrix of thunk blocks.
- Phase D note: when refinement produces Python `SubmatrixView` blocks, view-local `+`/`@` fall back to small dense evaluation (block-local only) to keep orchestration correct without global densification.
- Phase E completed: `.pycauset` save/load supports internal `BlockMatrix` via a manifest (`matrix_type=BLOCK`) that references child `.pycauset` files stored in a sibling directory (`<file>.blocks/`).
- Phase E behavior: saving a BlockMatrix evaluates `ThunkBlock` blocks blockwise (no global densify), persists realized results, and raises on stale thunks.
- Phase E hardening: thunk staleness pins include leaf blocks (and view sources) so in-place leaf mutation with a `version` bumps stale deterministically; concurrency test ensures single-eval caching under threads.
- Phase E hardening: child references are validated deterministically (path safety + pinned child `payload_uuid`), and multi-file saves use staging/commit to reduce partial-update risk.
- Phase E coverage: nested, mixed-dtype, and view-block (SubmatrixView) persistence paths have round-trip tests.

- Phase F started: public constructor disambiguation and “once block, always block” routing are wired.
- Phase F started: mixed-operand operator behavior (`dense + block`, `dense @ block`) defers to BlockMatrix via NotImplemented fallback in native operator bindings.
- Phase F started: leaf `matmul` inside thunk evaluation routes through the public compute boundary for native matrices, preserving property-aware dispatch.
- Phase F started: IO accelerator hooks are exposed (best-effort prefetch/discard) with trace validation support.
- Phase F started: integration tests cover operator fallback and device routing for block addition when CUDA is active (with CPU fallback for unsupported dtypes).
- Phase F completed: elementwise block ops (`+`, `-`, `*`, `/`) are thunked and partition-aligned, with IO prefetch/discard hooks.
- Phase F completed: integration tests cover mixed-operand fallbacks (`dense op block`) and device routing expectations for `+`/`-` (GPU when supported, CPU otherwise) plus CPU-only guarantees for `*`/`/` under CUDA.
- Phase G completed: documentation footprint added per Documentation Protocol — public guides (Matrix Guide, R1 linalg/storage), API reference (`pycauset.matrix`, `pycauset.matmul`, `pycauset.save`, `pycauset.load`), and internals (Block Matrices) now cover construction rules, slicing, triggers/non-triggers, refinement, staleness, persistence sidecars, and device routing expectations.
- Phase H completed: hardening tests added for complex matmul vs dense (CPU fallback with guarded skip when complex matmul/persistence unsupported), float16 matmul vs dense, many-small-block matmul vs dense, mixed-dtype add vs dense, nested complex matmul+persistence (guarded skip), thunk concurrency single-eval locking, and `set_block` staleness invalidation; suite is green with expected skips only.

Phase E policy choices (locked for this implementation pass):

- Sidecar directory naming: use `str(path) + ".blocks"` (example: `bm.pycauset` → `bm.pycauset.blocks/`).
- Overwrite cleanup: best-effort delete only deterministic child block files matching `block_r*_c*.pycauset` in the sidecar directory; do not wipe unrelated files.
- Stale thunk on save: saving must raise a deterministic stale-thunk error (no silent recompute).
- Persisted snapshot integrity: each manifest child entry pins the child file's `payload_uuid`; load validates this and errors if a child file was replaced/rewritten (prevents silent mixed-snapshot loads).
- Container-level cached-derived: no block-matrix-level `trace/determinant/norm/sum/...` caching is defined for R1 internal `BlockMatrix`; cached-derived persistence remains per-child (leaf matrices) under the existing signature rules.
- View persistence (R1 internal policy): if a BlockMatrix contains `SubmatrixView` blocks (e.g., from `refine_partitions`), saving materializes each view **block-locally** by copying elements into a small NumPy array (with dtype inferred from the view’s source type) and then converting it via `native.asarray`; there is no multi-file "view reference" format in R1.
- Multi-file save hardening (best-effort): BlockMatrix saves stage managed child files (and nested child `.blocks/` trees) in a temporary staging directory and then commit via replace/rename. This reduces partial updates; if a crash still occurs mid-commit, `payload_uuid` pinning ensures loads fail deterministically rather than silently mixing snapshots.

**What is next (NEXT):**

- Optional follow-ups: perf guardrails or stress/block-count overhead checks, plus CLI/bench hooks if product asks; otherwise plan is complete for R1 scope.

**Blocked / deferred:**

- Full symbolic lazy evaluation / fusion across arbitrary expression graphs is deferred.
- GPU kernels for block-aware ops are deferred; leaf ops must still route via AutoSolver/ComputeDevice.
*** Block-matrix-aware slicing completed (integrated into guides/API/internals); no longer deferred.

---

## Phased execution plan (docs/testing baked in)

- **Phase A — Contract lock (planning):** finalize semantics for evaluation triggers, caching/versioning, dtype accumulation, partition refinement, mutation, and manifest shape. Exit: written acceptance criteria + updated risk/open questions if any. Docs: update this plan + roadmap status.
- **Phase B — Core types & validation:** implement `BlockMatrix` container (partitions, lookup, mixed dtype metadata), construction validation, structural `repr`/`str`. Exit: construction/indexing unit tests + structure-only printing tests.
- **Phase C — Views & partition refinement:** deliver `SubmatrixView` (no-copy), block-grid refinement for mismatched partitions, tiled views across blocks, scalar/transpose propagation. Exit: view composition tests (dense + block), refinement correctness tests, deterministic error cases for unsupported views.
- **Phase D — Ops orchestration & thunks:** elementwise ops and block matmul orchestration with deterministic per-block accumulator dtype, thunk creation, evaluation triggers, and snapshot/version checks. Exit: lazy-eval tests (triggers/non-triggers), dtype accumulator tests, stale-thunk error tests, and CPU routing for complex blocks; add trace tags for observability.
- **Phase E — Persistence & caching:** manifest save/load for nested blocks, temp handling for thunks, cache invalidation on `set_block`/child mutation, single-eval locking. Exit: save/load round-trip tests (nested, mixed dtype), cache invalidation tests, concurrent thunk evaluation test.
- **Phase F — Integration (properties/device/IO):** ensure “once block, always block” behavior with AutoSolver boundary, property metadata pass-through, IO accelerator prefetch/discard hooks, and CPU fallback for CUDA gaps. Exit: integration tests covering property propagation, device routing expectations, and IO prefetch/discard trace validation.
- **Phase G — Documentation:** publish API + internals per Documentation Protocol (construction rules, triggers, manifest schema, versioning/staleness, device routing expectations, printing). Exit: docs merged + updated roadmap status.
- **Phase H — Testing & hardening:** expand coverage (complex blocks, float16, deep nesting), stress/block-count overhead checks, and CLI/bench hooks if needed. Exit: test suite green, known gaps documented; optional perf guardrails noted.

---

## Phase A — Contract lock (COMPLETED 2025-12-21)

This section freezes the semantics and interfaces so Phase B+ can implement without re-litigating contracts.

### A.1 Locked decisions

1) **User entrypoint (constructor) is `pycauset.matrix(...)` (extended, not replaced).**

The current `pycauset.matrix(source, dtype=None, **kwargs)` is a numeric data constructor (1D→vector, 2D→dense matrix).

Block-matrix construction will be added without breaking existing behavior:

- If `source` is 2D and **every element is a `MatrixBase`-like object**, treat `source` as a block grid and construct a `BlockMatrix`.
- If `source` is 2D and **no elements are matrices** (numeric scalars / NumPy scalars), keep current dense behavior.
- If `source` is 2D and **mixes matrices and scalars**, raise a deterministic `TypeError` (no implicit lifting / no implicit densification).

Rationale: avoids ambiguity with the existing numeric constructor while enabling the target syntax.

2) **Block-level access is explicit, element access remains elementwise.**

- `M[i, j]` returns an element.
- Block access is via `get_block(r, c)` / `set_block(r, c, block)` with `r,c` as block-grid indices.

3) **No silent densification.**

- Any unsupported view/refinement case must raise a deterministic error.
- There is no fallback that materializes a whole dense matrix “just to make it work”.

4) **Semi-lazy matmul outputs thunk per output block, with fixed determinism rules.**

- Thunks are evaluated only on triggers (element access, dense export, persistence, or leaf-compute boundary).
- Per-block accumulation dtype follows the deterministic metadata-only rule already defined in this plan.
- Summation order over `k` is fixed.

5) **Staleness semantics are snapshot-at-creation (default).**

- Thunks capture input references + versions.
- If any input version differs on evaluation/cache hit, raise a deterministic “stale thunk” error (do not silently recompute).

6) **Persistence shape is manifest + referenced children (nestable).**

- A saved block matrix stores a manifest in `.pycauset` metadata that describes the grid topology and references child matrices.
- Thunk blocks are evaluated **blockwise** for save (no global densify).
- If a referenced child is temporary, saving persists it to a stable child file first and then references it.

### A.2 Acceptance criteria (Phase A exit)

- The constructor disambiguation rules for `pycauset.matrix(...)` are explicit (numeric vs block-grid vs mixed error).
- Minimum public-facing block API is frozen: `get_block`, `set_block`, `block_rows`, `block_cols`, `row_partitions`, `col_partitions`.
- Evaluation triggers and non-triggers are frozen (especially: `repr/str` and metadata queries must never evaluate thunks).
- “No silent densify” rule is frozen (unsupported views/refinements error deterministically).
- Staleness semantics are frozen (snapshot-at-creation; stale access errors).
- Persistence intent is frozen (manifest + referenced child matrices; thunk save is blockwise).

---

## 0) Purpose and scope

Block matrices are a core storage format in PyCauset.

Primary goals:

- **Storage-first:** constructing/saving a block matrix must not write an expanded dense buffer.
- **Heterogeneous dtype:** a block matrix may contain blocks of different dtypes (including nested block matrices).
- **Infinitely nestable:** blocks may themselves be block matrices, recursively.
- **Frontend transparency:** a block matrix behaves like a matrix from the public API:
  - element indexing yields elements, not blocks,
  - matmul/elementwise ops “just work” under correct dimension rules.
- **Single dispatch boundary:** leaf computations route via AutoSolver/ComputeDevice.

Non-goals (for now):

- General-purpose lazy expression graphs across all operations.
- Implicit densification as a fallback for unsupported block/view cases.
- Perfect performance for deeply nested blocks (correctness first; inefficiency is acceptable).

---

## 1) User-facing construction and behavior

### 1.1 Construction

Target syntax:

- Example:

```python
M = pycauset.matrix([
  [A, B],
  [C, D],
])
```

where `A,B,C,D` are matrices (including block matrices).

Validation:

- All blocks in a **block-row** must have the same number of rows.
- All blocks in a **block-col** must have the same number of cols.
- Total shape is `(sum block-row heights, sum block-col widths)`.

### 1.2 Indexing semantics

- `M[i, j]` returns a scalar element.
- `M[i0:i1, j0:j1]` slicing is deferred to the broader indexing plan in the [R1_LINALG plan](R1_LINALG_PLAN.md); block work requires an internal view node.

### 1.3 Block replacement API

Block-level access is explicit (not via `__getitem__`).

Minimum API:

- `get_block(r, c) -> MatrixBase`
- `set_block(r, c, block: MatrixBase) -> None`

`r,c` are **block indices** into the current block grid (not element indices):

- `r` is the block-row index in `[0, block_rows)`
- `c` is the block-col index in `[0, block_cols)`

Required introspection API (minimal):

- `block_rows` and `block_cols` (or equivalent methods)
- `row_partitions` and `col_partitions` returning the boundary arrays (prefix sums) used by the block grid
  - Example: `row_partitions == [0, r1, r2, ..., rows]`
  - Example: `col_partitions == [0, c1, c2, ..., cols]`

`set_block` must validate that the replacement block matches the existing block-row height and block-col width.

Optional (nice-to-have) helpers for debugging:

- `block_shape(r, c) -> (rows, cols)`
- `block_dtype(r, c) -> DataType`

### 1.4 Visualization / printing (must not trigger evaluation)

Block matrices are primarily about structure. Default printing must emphasize structure without forcing computation.

Required behavior:

- `repr(M)` / `str(M)` prints a **structure summary** and never evaluates thunks.
- The summary includes:
  - overall shape,
  - block grid size (block_rows × block_cols),
  - row/col partitions,
  - per-block descriptors: shape, dtype, and kind (`leaf`, `view`, `thunk`, `block`).

Recommended formatting (human-scannable):

- 1-line header: `BlockMatrix(shape=(...), grid=RxC, dtype=MIXED)`
- Then an ASCII grid of block descriptors, truncated by `max_depth` and `max_blocks`.

Explicitly opt-in value printing (may evaluate):

- If we add a helper like `M.to_dense_string(...)` or `M.materialize()` later, it must be clearly named and documented as a trigger.



---

## 2) Core internal representation

### 2.1 Internal node types

`BlockMatrix : MatrixBase` (internal; not necessarily public as `pc.BlockMatrix`):

- stores `blocks[r][c]` as references (shared ownership),
- stores `row_offsets[]` and `col_offsets[]` prefix sums,
- supports nesting (a block can be another `BlockMatrix`).

`SubmatrixView : MatrixBase` (required):

- represents a rectangular view into an existing matrix without copying,
- supports views into dense matrices and block matrices (views can compose).

`ThunkBlock` / `DeferredBlock` (internal helper; may be a MatrixBase subclass or a block wrapper):

- represents a deferred computation that produces a concrete MatrixBase on evaluation,
- caches evaluated result.

### 2.2 Dtype model (heterogeneous containers)

Block matrices do **not** have a single dtype.

- Introduce `DataType::MIXED` (or equivalent) for heterogeneous containers.
- Each child block retains its own dtype.

Element reads:

- `get_element_as_double(i,j)` / `get_element_as_complex(i,j)` are defined by delegating into the appropriate block.

Structured view scope (initial):

- `SubmatrixView` supports dense matrices and block matrices first; other structured types must raise a deterministic “view not supported yet” error until explicitly added.

---

## 3) Operation contracts on block matrices

### 3.1 “Once block, always block” (default)

If any operand is a block matrix, the default result is a block matrix (unless explicitly materialized).

Rationale:

- preserves storage wins (identity/zero/diagonal blocks remain cheap),
- avoids global promotion/densification.

### 3.2 Elementwise ops (`+`, `-`, `*`, `/`)

Contract:

- Operate blockwise.
- If mixing `BlockMatrix` with a non-block matrix, partition the non-block operand into matching `SubmatrixView` tiles.
- Output block dtypes are computed per-block using the promotion rules for that specific op on the specific operand dtypes.

No silent densification:

- If a required view cannot be represented (e.g., unsupported structure), raise a deterministic error.

### 3.3 Matmul (`@`)

Contract:

- Block matmul follows standard block multiplication:

  For block grids `A[i,k]` and `B[k,j]`:

  $$C_{ij} = \sum_k A_{ik} @ B_{kj}$$

- Result is a `BlockMatrix` of output blocks.

#### Semi-lazy evaluation 

We use **semi-lazy** evaluation for block matmul outputs:

- `C` is returned immediately as a `BlockMatrix` of thunks.
- Each thunk represents one `C_ij` block.
- The thunk is evaluated on-demand and then cached.

##### Evaluation triggers (exactly when thunks run)

A thunk for an output block `C_ij` **must not** run “in the background”. It only runs when an API call requires numeric contents of that block.

**Triggers (must evaluate the minimal required blocks):**

- Element access (`C[i, j]`) evaluates the unique block that contains `(i, j)`.
- Any operation that requires dense contents of `C` evaluates all blocks:
  - conversion/export (`np.asarray(C)`, `pycauset.to_numpy(C)`),
  - any future `materialize()` API.
- Persistence (`C.save(path)`) evaluates all thunk blocks, but must do so **blockwise** (materialize each block into a child matrix file; never densify the full matrix into one giant buffer).
- Passing a thunk block across the leaf compute boundary (AutoSolver/ComputeDevice) evaluates it first.

**Non-triggers (must not evaluate):**

- `C.shape`, `C.rows`, `C.cols`, and any partition metadata queries.
- `C.dtype` / “container dtype” queries for a mixed container.
- `repr(C)` / `str(C)` / debug printing (may report “thunked” status).
- `C.get_block(r,c)` (returns a handle that may still be thunked; evaluation is triggered only when numeric contents are required).

Determinism:

- Reduction order over `k` is fixed.
- Dtype decisions for accumulation are deterministic.

Local promotion (unavoidable):

- Even with heterogeneous container dtypes, a single output block `C_ij` may need local promotion while accumulating `Σ_k`.
- This is local to that block; it does not force a global result dtype.

##### Deterministic dtype accumulation rule for $\sum_k$ inside one output block

Each output block `C_ij` is computed as a deterministic fold over terms `T_k = A_ik @ B_kj`.

**Rule (deterministic, metadata-only):**

1) For each `k`, compute the **term dtype** `dtype(T_k)` using the existing matmul dtype rules (the same rules used by the non-block `@`). This is determined from operand dtypes/structures and does not require evaluation.

2) Compute the **accumulator dtype** `dtype_acc` by folding the existing addition dtype rule over the sequence of term dtypes in increasing `k`:

$$dtype\_acc = fold\_k\big( add\_result\_dtype,\ dtype(T_0), dtype(T_1), \ldots \big)$$

3) During numeric evaluation, each term `T_k` is accumulated into an accumulator buffer of type `dtype_acc` (casting each `T_k` to `dtype_acc` if needed) using a fixed `k` order.

**Notes:**

- This keeps the container heterogeneous: `dtype_acc` is per-output-block.
- If term dtypes include both real and complex, `dtype_acc` is complex (per existing promotion rules).
- If term dtypes include multiple float widths, `dtype_acc` is the widest among them (per existing promotion rules).
- Integer-only accumulation follows the existing integer promotion behavior for `+` (we are not adding a new “safe integer accumulator” mode in R1).

#### Partition mismatch handling

If block partitions are not aligned:

- Compute the common refinement of boundaries and use `SubmatrixView` to split blocks without copying.
- If refinement cannot be represented without copying, raise an error (do not silently densify).

##### Partition mismatch refinement via `SubmatrixView` (no silent densify)

We must support the common case where `A` and `B` have different block boundaries, while still guaranteeing no implicit densification.

**Matmul refinement (core requirement):**

- Let `A` have row boundaries `RA` and col boundaries `CA`.
- Let `B` have row boundaries `RB` and col boundaries `CB`.

For `C = A @ B`, the shared dimension boundaries must align in the refined view. We compute:

- `K = sort(unique(CA ∪ RB))` as the refinement boundaries for the shared axis.

Then, for each output block `(i, j)` defined by row interval `RA[i..i+1]` and col interval `CB[j..j+1]`, the sum is taken over `k` intervals defined by `K`:

- `A_ik_view = SubmatrixView(A, rows=RA_i, cols=K_k)`
- `B_kj_view = SubmatrixView(B, rows=K_k, cols=CB_j)`

These views must be representable without copying for all participating blocks.

**Elementwise refinement (for completeness):**

- For `A ⊙ B` (⊙ ∈ {+,-,*,/}), compute refinement boundaries on both axes:
  - `R = sort(unique(RA ∪ RB))`
  - `C = sort(unique(CA ∪ CB))`
- Tile both operands into matching `SubmatrixView` blocks.

**Hard rule:**

- If any required `SubmatrixView` is not representable under a matrix’s structure constraints (e.g., a square-only structured type that cannot form an arbitrary rectangular view), the operation must raise a deterministic error.
- There is no “fallback densify” behavior.

---

## 4) Mutation/versioning and caching rules (critical)

Lazy thunks + mutation can produce silent wrong answers unless we define this explicitly.

### 4.1 Default semantics: snapshot for thunks

R1 default (locked in):

- Thunks capture a snapshot of inputs (by reference + versioning).
- If any input block changes after thunk creation, cached results are invalidated or evaluation is forbidden until re-derived.

Recomputation policy:

- If serving a request would require more than O(1) work relative to cached state, do not auto-recompute; stale accesses raise a deterministic error. In R1, all stale thunk/cache hits fall under this rule and must error.

Implementation direction:

- Add per-object monotonic `version` counters that increment on mutation.
- Thunks store the input versions they were derived from.
- On evaluation/cache hit, verify versions match.

Clarification (R1 choice — explicit):

- This is **snapshot-at-creation semantics** for lazy results: a thunk is only valid for the specific versions of its inputs that existed when the thunk was created.
- If an input’s version differs, the thunk must raise a deterministic “stale thunk” error rather than silently recomputing with new values.

This avoids time-dependent results and mixed-snapshot matrices.

Future (explicit opt-in, not default):

- We may add a “live” policy where stale thunks automatically recompute against current inputs, but it must be user-controlled (never implicit) because it changes determinism and interacts badly with partial caching.

### 4.2 Caching policy

Pitfall: unbounded caching can explode disk/memory.

Minimum viable policy:

**What is cached:**

- The evaluated numeric result of each thunk block (`C_ij`) as a concrete `MatrixBase` instance.

**Where it is cached:**

- Cached blocks are stored in-memory as references, and their backing storage is a normal matrix backing (typically a temp `.pycauset` file) managed by the existing temp lifecycle.

**Invalidation:**

- Any `set_block` on a `BlockMatrix` invalidates all cached/thunked blocks owned by that `BlockMatrix` (increment the parent version; cached blocks become inaccessible).
- Any mutation of a referenced child matrix invalidates any thunk (and any cached block) that depends on it via version mismatch checks.

**Eviction:**

- No eviction policy is required for R1 beyond the existing temp cleanup. (We avoid introducing new user-facing cache knobs in R1.)

Later enhancements (SRP/IO):

- LRU eviction, memory thresholds, explicit `cache_policy` knobs.

### 4.3 Mutation semantics

We need explicit semantics for both (a) replacing blocks and (b) mutating matrices that are referenced as blocks.

#### `set_block(r, c, block)`

- `set_block` is a mutation of the `BlockMatrix` container.
- It must validate the replacement block’s shape against the existing block-row and block-col sizes.
- It increments the container’s `version` and invalidates any cached/thunked blocks owned by the container.
- It does not mutate the old or new child blocks.

#### Edits to referenced child matrices

If a child matrix `A` is referenced by one or more `BlockMatrix` containers and the user mutates `A` (element edits, in-place ops, etc.):

- `A.version` increments.
- Any thunk that captured `A` at an older version becomes **stale**.
- Accessing a stale thunk (or a cached block derived from a stale thunk) must raise a deterministic error (not recompute). This is the R1 default.

Rationale:

- Prevents partially materialized results that mix old and new input states.
- Keeps “lazy” behavior from becoming time-dependent.

---

## 5) Persistence: reference-save manifests (infinitely nestable)

### 5.1 Runtime assumption

During a run, backing `.pycauset` files do not move or disappear.

### 5.2 Save semantics

When a user calls `.save(path)` on a block matrix:

- Write a **manifest** that records:
  - block grid topology,
  - row/col partition sizes,
  - for each block: reference to the child’s backing file + block metadata (transpose/conjugate/scalar/etc),
  - recursion for nested blocks.

Format requirement:

- Manifest is a binary header + records inside the `.pycauset` file; JSON/zip formats are removed and must not be referenced in code or docs.

The manifest does **not** store expanded dense elements.

Handling temporaries:

- If a referenced child is temporary, saving must first persist it to a stable child file (copy storage) and then reference that.

### 5.3 Load semantics

- Load reads the manifest, reconstructs the `BlockMatrix` structure, and recursively loads child blocks.

---

## 6) Integration points (single dispatch boundary preserved)

- `BlockMatrix` orchestration decomposes operations into leaf ops on non-block matrices.
- Leaf ops always route via AutoSolver/ComputeDevice.

This keeps OpenBLAS/CUDA integration straightforward:

- They accelerate the leaf `matmul/add/...` kernels.
- Block orchestration decides *which* leaf ops to run and *when* (via thunks).

### 6.1 Alignment with Compute Architecture

This plan is designed to remain compatible with `documentation/internals/Compute Architecture.md`:

- All heavy numerical work still goes through `ComputeContext::instance().get_device()`.
- AutoSolver remains the routing point that decides CPU vs GPU based on size and capabilities.

#### Device capability reality (important)

CUDA matmul is not a universal backend today:

- `CudaDevice::matmul` supports dense `float32`/`float64` matrices.
- Complex matmul is supported on CPU (`cpu.matmul.c32`, `cpu.matmul.c64`, plus `cpu.matmul.cf16_fallback`) but is not supported on CUDA today.

Therefore:

- Block matrices containing complex blocks must still work, but AutoSolver must route those leaf ops to CPU.
- Mixed-dtype block containers are fine because routing happens per leaf op, not per container.

### 6.2 IO Accelerator compatibility (prefetch/discard)

To stay compatible with the IO acceleration layer:

- Before evaluating a thunk block, the orchestrator should prefetch the required input blocks/views (where available) via each matrix’s accelerator interface.
- After a thunk block is evaluated and its intermediate temporaries are no longer needed, the orchestrator should allow (or request) discard of those temporaries.

This preserves the existing “minimize page faults” strategy for out-of-core workloads.

### 6.3 Views and BLAS/GPU friendliness (without copying)

`SubmatrixView` must not force a copy. For performance (later), it should expose an optional “dense window” representation when possible:

- pointer/offset/leading-dimension view into a dense backing,
- otherwise fall back to the generic `get_element_as_*` path on CPU.

GPU support for views can be added later if/when CUDA kernels accept strides/offsets.

---

## 7) Testing strategy (minimum)

- Construction:
  - valid block grids build without densification.
  - invalid grids raise deterministic errors.
- Indexing:
  - element reads match dense equivalent.
- Mixed dtype:
  - block containers can hold multiple dtypes.
  - elementwise ops produce per-block dtype results.
- Semi-lazy matmul:
  - no compute until triggered (verify via kernel trace tags),
  - deterministic results,
  - cache hit behavior.
- Persistence:
  - save/load round-trip for nested manifests.

Additions (to catch subtle bugs early):

- Complex blocks (CPU fallback): ensure complex32/complex64/complex-f16 blocks round-trip through elementwise ops and matmul without densifying the whole matrix.
- Printing: `repr`/`str` must not evaluate thunks (verify via trace tags).

---

## 8) Risk register

- Snapshot/versioning complexity: must avoid silent stale caches.
- Excessive overhead for many small blocks: mitigated by later scheduling/materialization knobs.
- Deep nesting inefficiency: accepted for completeness.
- CUDA capability gaps: complex blocks and some view/shape cases will route to CPU until CUDA support expands.
- Thread-safety: concurrent evaluation of the same thunk block must not corrupt caches (R1 uses single-eval locking; see below).
- View correctness: views over nested blocks that cross block boundaries are subtle; ensure we represent them as structured compositions of smaller views, not as copies.
- Scalar/transpose flags: ensure `SubmatrixView` and manifest metadata preserve scalar multipliers and transpose status without changing numeric meaning.

---

## 9) Open questions (to decide early)

1) Exact API surface for block operations (`get_block`/`set_block`, `materialize`, cache controls).
2) Whether `SubmatrixView` supports all structured matrices or only dense initially.
3) Manifest format is locked to the binary header + records inside `.pycauset`; JSON/zip are deprecated and removed.

R1 decisions (locked in):

4) Concurrency semantics for thunks: use single-eval locking per thunk (e.g., `std::mutex` + double-checked caching or `std::once_flag`). If multiple threads request the same thunked block, only one computes and installs the cached result; other threads wait and then reuse the cached block.

5) `SubmatrixView` spanning multiple blocks is allowed: represent it as a `BlockMatrix` of `SubmatrixView` tiles (a “tiled view”), with no copying and no densification.
  - Note: such a tiled view may be `DataType::MIXED` if it spans blocks with different dtypes. Element access remains well-defined; dense materialization chooses a target dtype using the existing dtype promotion rules.

---

## 10) Documentation requirements (must be thorough)

We need both API docs and internals docs so future work (I/O, CPU, GPU) can safely optimize without breaking semantics.

### 10.1 Public/API documentation

- `pycauset.matrix(...)` block-grid construction rules and validation errors.
- Block matrix behavior expectations:
  - element indexing returns elements,
  - block replacement only via explicit API (`get_block`/`set_block`),
  - “once block, always block” default.
- Evaluation triggers (what forces computation) and non-triggers.
- Save/load semantics for manifests (what is stored, what is not).
- Device routing expectations (AutoSolver may run different blocks on CPU vs GPU depending on dtype/shape/capability).
- Printing/visualization behavior (structure summary by default; no implicit evaluation).

### 10.2 Internals documentation

- Data model: `BlockMatrix`, `SubmatrixView`, `ThunkBlock`.
- Versioning + staleness semantics (snapshot-at-creation; stale thunks error).
- Caching rules + lifecycle of temp files.
- Partition refinement algorithm and “no silent densify” rule.
- Manifest format: schema, examples, and compatibility rules for future extensions.
- Complex-number storage notes:
  - complex32/complex64 dense matrices store `std::complex<T>` values,
  - ComplexFloat16 matrices may store real/imag planes; views and thunks must treat them consistently.

---

## 11) Agent handoff: implementation checklist (for the next AI agent)

This section is intentionally explicit to reduce the chance of subtle bugs.

### 11.1 Core correctness checklist

- Implement `BlockMatrix` shape, partitions, and fast element-to-block lookup (binary search on `row_partitions`/`col_partitions`).
- Implement `SubmatrixView` with strict no-copy semantics and correct composition over:
  - transpose flags,
  - scalar multipliers,
  - nested block matrices.
- Implement `ThunkBlock` evaluation + caching with version checks:
  - snapshot input versions at creation,
  - error on stale access.

### 11.2 Compute boundary checklist (must match Compute Architecture)

- Orchestrator decomposes ops into leaf ops and calls AutoSolver/ComputeDevice.
- Respect device capability gating:
  - complex blocks must route to CPU today,
  - CUDA paths are limited to supported dtypes/shapes.

### 11.3 Determinism checklist

- Block matmul thunk evaluation must use a fixed `k` order.
- Accumulator dtype for each output block must be determined from dtype metadata only (no data-dependent decisions).
- Avoid parallel reductions that change summation order unless explicitly documented.

### 11.4 Persistence checklist

- Manifest save/load must preserve nesting and references.
- Saving must not write a single expanded dense buffer for the whole block matrix.
- Thunk blocks must be saved as stable child matrices (evaluate per-block), then referenced.

### 11.5 Debug/visualization checklist

- `repr`/`str` must never evaluate thunks.
- Provide readable structural summaries (shape, partitions, per-block metadata).

### 11.6 “Don’t get trapped” pitfalls

- Do not accidentally materialize views to make BLAS happy (violates storage-first).
- Be explicit about complex behavior:
  - CPU supports complex matmul; CUDA does not today.
  - ComplexFloat16 uses split planes; views must handle both planes consistently.
- Ensure partial caching cannot produce mixed-snapshot results (stale access must error).
