# R1_BLOCKMATRIX Plan — Block Matrices + Heterogeneous Dtypes (Storage-First)

**Status:** Active (drafted 2025-12-17)

This plan is the canonical source of truth for block matrices in Release 1.

## Progress snapshot (MUST keep updated)

**Last updated:** 2025-12-17

**Current phase step:** Design lock (contracts + persistence format + thunk model)

**What is done (DONE):**

- Block matrix requirements discussed and captured (heterogeneous dtypes, reference-save manifests, semi-lazy matmul thunks).
- Roadmap node `R1_BLOCKMATRIX` created; block work moved out of R1_LINALG.

**What is next (NEXT):**

- Formalize contracts: dtype semantics, mutation/versioning, evaluation triggers, caching policy, and partition refinement rules.
- Implement internal node types (`BlockMatrix`, `SubmatrixView`, `ThunkBlock`/equivalent) and wire persistence manifests.

**Blocked / deferred:**

- Full symbolic lazy evaluation / fusion across arbitrary expression graphs is deferred.
- GPU kernels for block-aware ops are deferred; leaf ops must still route via AutoSolver/ComputeDevice.

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

- `pycauset.matrix([ [A, B], [C, D] ])` where `A,B,C,D` are matrices (including block matrices).

Validation:

- All blocks in a **block-row** must have the same number of rows.
- All blocks in a **block-col** must have the same number of cols.
- Total shape is `(sum block-row heights, sum block-col widths)`.

### 1.2 Indexing semantics

- `M[i, j]` returns a scalar element.
- `M[i0:i1, j0:j1]` slicing is deferred to the broader indexing plan; block work requires an internal view node.

### 1.3 Block replacement API

Block-level access is explicit (not via `__getitem__`).

Minimum API:

- `get_block(r, c) -> MatrixBase`
- `set_block(r, c, block: MatrixBase) -> None`

`set_block` must validate that the replacement block matches the existing block-row height and block-col width.

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

#### Semi-lazy evaluation (decision)

We use **semi-lazy** evaluation for block matmul outputs:

- `C` is returned immediately as a `BlockMatrix` of thunks.
- Each thunk represents one `C_ij` block.
- The thunk is evaluated on-demand and then cached.

##### Evaluation triggers (exactly when thunks run)

A thunk for an output block `C_ij` **must not** run “in the background”. It only runs when an API call requires numeric contents of that block.

**Triggers (must evaluate the minimal required blocks):**

- Element access (`C[i, j]`) evaluates the unique block that contains `(i, j)`.
- Any operation that requires dense contents of `C` evaluates all blocks:
  - conversion/export (`np.array(C)`, `pycauset.asarray(C)`),
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

---

## 8) Risk register

- Snapshot/versioning complexity: must avoid silent stale caches.
- Excessive overhead for many small blocks: mitigated by later scheduling/materialization knobs.
- Deep nesting inefficiency: accepted for completeness.

---

## 9) Open questions (to decide early)

1) Exact API surface for block operations (`get_block`/`set_block`, `materialize`, cache controls).
2) Whether `SubmatrixView` supports all structured matrices or only dense initially.
3) Exact manifest format (JSON vs custom binary header + records). Recommended: small custom binary for speed + robustness.
