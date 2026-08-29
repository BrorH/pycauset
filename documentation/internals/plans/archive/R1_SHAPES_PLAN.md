# R1_SHAPES Plan — NxM Matrices Across The System (End-to-End)

**Status:** Complete (Phase 0 complete; Phase 1 complete; Phase 2 complete; Phase 3 complete)

## 0) Purpose and scope

R1_SHAPES is the first “shape-lifts-everything” milestone. The goal is to remove square-only assumptions so later optimization and feature work does not require rewrites.

This plan is written to match repo invariants:

- **Scale-first:** do not introduce paths that materialize large out-of-core data.
- **Tiered storage:** RAM vs disk is an implementation detail; API must stay stable.
- **Lazy evaluation / metadata-first:** prefer flipping metadata to touching data.
- **Centralize policy:** avoid scattered shape rules.
- **Warnings vs exceptions:** shape mismatches and unsupported shapes must raise deterministically.

Documentation requirement (always on): any new public function/method/class added or behavior changed must be documented per `documentation/project/protocols/Documentation Protocol.md`.

Authoritative higher-level context:

- Project roadmap node: `documentation/internals/plans/TODO.md` → **R1_SHAPES**.
- Square-only map: `documentation/dev/Square-only Assumptions.md`.
- Warning/exception rules: `documentation/dev/Warnings & Exceptions.md`.
- Philosophy mantra: `documentation/project/Philosophy.md`.

> Documentation note:
>
> This file is a planning/spec artifact. User-visible R1 behavior is documented in:
>
> - [[guides/release1/shapes.md|Release 1: Shapes (what shipped)]]
> - [[guides/NxM Support.md|NxM Support Status]]
> - [[guides/Matrix Guide.md|Matrix Guide]] and [[guides/Vector Guide.md|Vector Guide]]
> - Factory API pages (e.g., [[docs/functions/pycauset.matrix.md|pycauset.matrix]], [[docs/functions/pycauset.zeros.md|pycauset.zeros]])

Non-goals (explicit):

- No N-D tensors. Only vectors and matrices.
- No new “physics” features.
- Phase 1 does **not** require making every operation NxM-capable (square-only ops may and should fail fast).

**Discoverability requirement:** we must maintain an easily discoverable list of which operations are currently disabled or restricted for NxM (and why), so it can be picked up later in the linear algebra TODO step.

---

## 1) Decisions already made 

These are treated as requirements for this plan.

### 1.1 Public naming conventions (Python)

- Lower-case naming is the forward-facing standard for user-level factories, in alignment with NumPy.
- As part of this milestone, the uppercase factories will be **purged** from the public API surface (no “deprecated” aliases, no warnings).

Naming policy going forward:

- User-level module functions and factories should be lower-case.
- Concrete native types remain PascalCase.

Required public factories:

- `pycauset.matrix(source, dtype=None, ...)`
- `pycauset.vector(source, dtype=None, ...)`

Related creators (also lower-case):

- `pycauset.zeros(shape, dtype=..., ...)`
- `pycauset.ones(shape, dtype=..., ...)`
- `pycauset.empty(shape, dtype=..., ...)`

Filling:

- Provide an analog to NumPy’s `ndarray.fill(...)` (method-style), for both vectors and matrices.

Additional planned renames (governed by the NumPy alignment protocol):

- Ensure the lower-case causal constructor `pycauset.causal_matrix(...)` exists and is used in-repo.
- Keep the class `pycauset.CausalSet` as the canonical type; prefer the lower-case convenience constructor `pycauset.causet(...)`.

Governing policy: `documentation/project/protocols/NumPy Alignment Protocol.md`.

We should keep optimized native *classes* in PascalCase (e.g. `FloatMatrix`) because they represent concrete types rather than user-level constructors.

### 1.2 Matrix factory semantics (Python)

- `pycauset.matrix(...)` must behave like NumPy’s array constructor: it takes iterable data (vector-like or matrix-like).
- `pycauset.matrix((n, m))` must **not** be interpreted as shape. In NumPy, `np.array((n, m))` is a 1×2 array; we follow that mental model.

Allocation rule (NumPy-aligned):

- `pycauset.matrix(...)` and `pycauset.vector(...)` construct from data only.
- Shape-based allocation is done via `pycauset.zeros/ones/empty`.

Implication:

- `pycauset.matrix(10)` is treated like `np.array(10)` (scalar input). Since scalars/0D are out of scope, this must raise a deterministic exception.
- `pycauset.vector(10)` likewise should not be “allocate a length-10 vector”; use `pycauset.zeros((10,), dtype=...)` / `pycauset.empty((10,), dtype=...)`.
- For empty initialization by shape, introduce NumPy-like creators:
  - `pycauset.zeros((rows, cols), dtype=...)`

Also required now:

- `pycauset.ones((rows, cols), dtype=...)`
- `pycauset.empty((rows, cols), dtype=...)`

Vector allocation uses 1D shapes:

- `pycauset.zeros((n,), dtype=...)`
- `pycauset.ones((n,), dtype=...)`
- `pycauset.empty((n,), dtype=...)`

**Dtype requirement (important):** for `zeros/ones/empty`, `dtype` must be explicitly provided.

Notes:

- `empty` may contain arbitrary/uninitialized contents. This is an advanced-user tool; “junk values” are acceptable. This MUST be stated in the documentation

**Policy:** If input is not vector- or matrix-shaped, throw a deterministic exception.

### 1.3 Shape dimensionality constraints

- Only vectors (1D) and matrices (2D) are supported.
- Higher-rank nested sequences must be rejected.

Vector creation rule:

- `pycauset.matrix([1,2,3])` returns a **vector** (NumPy mental model: 1D input becomes 1D output).

This raises a design consideration: vectors and matrices are both “matrices” at the storage level (rows/cols + flags), but are distinct front-end types for ergonomics. The plan below preserves that split: distinct Python factories and Python-visible behaviors; shared backend representation.

### 1.4 `shape()` and `size()` semantics

`shape` must behave like NumPy:

- Matrices: `shape == (rows, cols)`
- Vectors: `shape == (n,)`

If there is not already a `shape()` method, implement it.

- Preferred: expose `shape` as an attribute/property (NumPy-like), and additionally provide `shape()` as a convenience method for parity with `size()`.

- `size()` must become NumPy-like: **total number of elements**.
  - For a matrix: `size = rows * cols`.
   - For a vector: `size = n`.

We do not introduce a replacement for the old “square dimension” concept (no new `dim()`/`n()` API). Code should use `rows()` / `cols()` everywhere.

This is a breaking semantic change and must be done carefully.

`__len__` must be NumPy-like:

- For matrices, `len(x) == x.shape[0]` (rows)
- For vectors, `len(x) == x.shape[0]` (n)

### 1.5 Persistence + transpose

- Persist **base dims + transpose flag** (and conjugation flag), not “normalized logical dims”.
- Transpose must be metadata-only (no full rewrite), consistent with the project’s lazy evaluation mantra.
- Vectors are a 1D frontend type, but use the same `(rows, cols)` + flags storage/persistence machinery.

Persistence format decision:

- Keep transform state (transpose/scalar/conjugation) in the typed metadata block.
- Do not introduce additional per-object headers beyond the `.pycauset` container header; view-state should remain metadata-only.

Vector representation (chosen: Option A / NumPy-like vectors):

- Frontend: `pycauset.vector(...)` creates a 1D vector with `shape == (n,)`, `size() == n`, `len(v) == n`.
- Transpose behavior: `v.T` is a no-op (returns a vector view/alias with identical shape).
- Backend/persistence: vectors are stored using canonical base dims `(rows=n, cols=1)` plus metadata. On load, the vector frontend is reconstructed (not a 2D matrix), so users never have to reason about `n×1` vs `1×n` for vectors.

Implication for Phase 2 (matmul/matvec semantics): vectors participate in multiplication using NumPy-compatible 1D rules; orientation is handled by operation rules rather than by exposing separate row-vs-column vector types.

### 1.6 Initial dtype scope

- Final target: all dense dtypes support NxM.
- Implementation strategy: start with **Float64 dense** to establish the pattern, then expand to other dense dtypes.

### 1.7 Zero-copy transpose

- Transpose of disk-backed objects must not duplicate storage; it should create a new object/view referencing the same underlying `.pycauset` data.

### 1.8 Square-only structures

- Strict validation: if a structure requires square (by our policy for Phase 1), it must throw on non-square.


### 1.9 Backward compatibility

- Backward compatibility is not a requirement. If we “deprecate”, we purge.

### 1.10 First success criteria

We care about all three:

- Rectangular `get/set` correctness.
- `np.asarray` / NumPy roundtrip shape correctness.
- Persistence save/load preserves `(rows, cols)` and transpose semantics.

### 1.11 Square-only ops behavior

- Ops that mathematically require square (inverse, det, eigen, etc.) must throw for non-square inputs.

---

## 2) Key architectural invariant to implement

### 2.1 Single source of truth for shape

Currently the engine has a split-brain:

- `PersistentObject` already stores `rows_` and `cols_`.
- `MatrixBase` stores `n_` and reports `rows()==cols()==size()==n_`.

For NxM, **we must make shape a single coherent concept** with a clear definition of:

- “base/storage” shape (what is laid out on disk/RAM)
- “logical/view” shape (what the user sees after transpose/conjugation flags)

Proposed invariant:

- `PersistentObject.rows_/cols_` represent **base/storage dims**.
- `is_transposed` flips the logical interpretation.
- `MatrixBase.rows()` / `cols()` return **logical dims**, derived from `PersistentObject` + flags.

Index mapping for dense row-major storage:

- Base layout is row-major with base stride = `base_cols`.
- For element reads/writes (`M[i, j]` and `M[i, j] = value`):
  - if `is_transposed`: swap `(i,j)` before mapping
  - `idx = i * base_cols + j`

This preserves the “transpose is a view flag” design and avoids touching data. But be careful about this implementation because any O(1) operation added to a frequently used method like set or get may accumulate.

Implementation note for performance (to keep hot paths hot):

- Provide a fast path when `is_transposed == false` to avoid extra branches/swaps in element access.
- Keep the mapping logic tiny (ideally inlineable) and avoid allocations.

### 2.2 Consequence: update `size()` everywhere

Once `size()` becomes total elements, any code that used `size()` as “dimension N” must be migrated to use `rows()` / `cols()`.

We should expect many call sites in:

- `src/math/*`
- compute routing heuristics that use `n_elements`
- Python helpers (`formatting`, `persistence`, coercion)

---

## 3) Work breakdown (phased)

### Phase 0 — Public API lowercase sweep (purge uppercase factories)

**Status:** Complete

**Goal:** make `pycauset.matrix` / `pycauset.vector` the canonical forward-facing factories, and remove PascalCase factory names from the public surface.

Clarification:

- “Purge” means removing PascalCase factory names from the module surface.
- The underlying implementation should be reused; in practice this should be mostly a mechanical rename/sweep (plus doc page renames and a few call-site adjustments where the old name was a class vs function).

Deliverables:

1. Implement `pycauset.matrix` and `pycauset.vector` in the Python facade.
2. Remove uppercase factories from exports and update all internal call sites accordingly.
3. Perform a repo-wide update of documentation, examples, tools, and tests.
4. Ensure docs index + class pages align with the final public names.

5. Add and adopt `documentation/project/protocols/NumPy Alignment Protocol.md` and obey it for future public surfaces.

6. Apply the protocol-driven renames planned in this milestone:
   - Ensure `pycauset.causal_matrix(...)` exists and is used in-repo.
   - Ensure `pycauset.causet(...)` exists and is used where appropriate.

Documentation specifics:

- Rename relevant docs pages so they match the final public symbol names.
- Update roamlinks/mkdocs navigation accordingly (noting the existing docs protocol guidance about `.` in filenames).

Acceptance criteria:

- `pycauset.matrix` and `pycauset.vector` exist and are used everywhere in-repo.
- `pycauset.causal_matrix` exists and is used everywhere in-repo.
- If we add `pycauset.causet(...)`, it exists and is used everywhere in-repo.
- Documentation pages and examples reference only the lower-case factories.

Acceptance checklist:

- All in-repo references have been migrated (tests, tools, benchmarks, docs, examples).
- Documentation updated per `documentation/project/protocols/Documentation Protocol.md` for each renamed public symbol.

Notes:

- This step is intentionally large and mechanical; it is worth doing early to avoid rewriting examples twice.
- Do not emit any messaging like “X is deprecated, use Y”. Old names simply stop existing.

### Phase 1 — Rectangular-safe dense objects end-to-end (Float64 first)

**Status:** Complete (validated 2025-12-17)

**Goal:** Rectangular dense Float64 matrix works correctly across allocation, indexing, transpose, NumPy interop, and persistence. Other ops may still be square-only.

Deliverables:

1. **C++ core:** Rectangular `DenseMatrix<double>`
   - Constructors accept `(rows, cols)`.
   - Storage allocates `rows*cols*sizeof(T)`.
   - Bounds checks use logical rows/cols.
   - Transpose toggles metadata and returns a view referencing the same mapper.

2. **C++ core:** Rectangular-aware `MatrixBase`
   - Remove `n_` as the authoritative dimension, or ensure it is not used as such.
   - Define `rows()/cols()/size()` semantics consistently.

3. **Factory + persistence plumbing (C++):**
   - `ObjectFactory::{create,load,clone}_matrix` must support `(rows, cols)` for dense.
   - For square-only structures: validate `rows==cols` and throw `std::invalid_argument`.

4. **Bindings:**
   - `shape` property must reflect logical dims.
   - `__array__` must allocate NumPy arrays of shape `(rows, cols)`.
   - NumPy import (`asarray` / dense-from-numpy) must accept non-square 2D arrays for dense.

5. **Python layer:**
   - `pycauset.matrix` must accept rectangular nested sequences and 2D numpy arrays.
   - Add `pycauset.zeros/ones/empty((rows, cols), dtype=...)` for shape-based matrix creation.
   - Add `pycauset.zeros/ones/empty((n,), dtype=...)` for shape-based vector creation.
   - Add `.fill(value)` for matrices and vectors.
   - Ensure `shape` and `shape()` exist (NumPy-like), and that `size()` is total elements.

Creator semantics (important):

- `zeros` and `ones` must perform a full write consistent with their meaning.
- `empty` may avoid initialization and therefore may contain arbitrary contents.
- For `zeros/ones/empty`, `dtype` must be explicitly provided. Document this (and why!) and throw exception if not passed.

6. **Python persistence:**
   - Save metadata must store explicit `rows` and `cols` based on shape (not `size()`).

7. **Tests:**
   - Rectangular creation + `get/set` edges.
   - Transpose flips logical shape without storage copy.
   - NumPy roundtrip preserves shape.
   - Persistence save/load preserves `(rows, cols)` + transpose flag.

Acceptance checklist:

- Can create a dense Float64 matrix from a 2D rectangular NumPy array; shape preserved.
- Can create a dense Float64 matrix via `pycauset.zeros((rows, cols), dtype=float64)`.
- Can create a dense Float64 vector via `pycauset.zeros((n,), dtype=float64)`.
- `get/set` works at edges for rectangular shapes.
- `T` produces a zero-copy view (no data duplication) with swapped logical shape.
- `pycauset.save` / `pycauset.load` roundtrip preserves shape and transpose.
- Vector persistence roundtrip preserves `shape == (n,)`.
- Square-only ops raise deterministic errors for non-square inputs.

Phase 1 completion record (2025-12-17):

- Native build + project-native unit tests passed (direct gtest executables produced under `build/Release`).
- Python test suite passed: `python -m pytest -q tests/python`.
- API + semantics validated end-to-end for rectangular dense numeric matrices:
   - `rows()/cols()` reflect logical dims (transpose is metadata-only),
   - `size()` is total elements (`rows * cols`),
   - NumPy `asarray` / `np.array(m)` roundtrip preserves shape,
   - persistence roundtrip preserves base dims + transpose flag.

### Phase 1.5 — Expand dense dtypes (mechanical)

**Status:** Complete (validated 2025-12-17)

Once Float64 is stable, extend the same rectangular-safe implementation to:

- float32, float16
- integer dense types
- complex dense types
- dense bit matrix (bit-packed) — requires “stride by cols”, not by `n_`

The intent is to reuse the same invariants and tests, with dtype-specific differences limited to:

- element size
- bit-packed stride computation
- complex scalar/conjugation safety

Acceptance checklist:

- The same rectangular creation + shape + persistence + transpose tests pass for each newly supported dtype.
- No dtype-specific path reintroduces square-only assumptions.

Phase 1.5 validation note (2025-12-17):

- Rectangular allocation + transpose-view shape were smoke-validated for: float16/32/64, int16/32, uint8/64, complex_float16/32/64.
- Dense `bool/bit` matrices support rectangular `(rows, cols)` allocation; bit-packed storage uses a stride derived from `cols`.

### Phase 2 — NxM operation rules (matmul/matvec/vecmat + elementwise)

**Status:** Complete (validated 2025-12-17)

Not the immediate coding target in Phase 1, but must be planned for:

- Matmul must support `NxM @ MxK -> NxK`.
- Elementwise ops must require shape equality.
- Matvec/vecmat must support conventional rules.

Python fallback in `pycauset._internal.ops.matmul` must be updated accordingly (or must raise a clear error if we intentionally disallow fallback for large problems).

Note, the operations need only be implemented at a sequential level. Optimization and parallelization belongs to another plan in TODO.md

**Deliverable (discoverability):** maintain a list of which ops are NxM-enabled vs restricted/square-only, in a location that is obvious from the docs index and/or the project TODO.

Phase 2 completion record (2025-12-17):

- Already implemented (spillover from Phase 1 work):
   - Matrix-matrix matmul follows the NxM rule (`(N,M) @ (M,K) -> (N,K)`) for the supported dense numeric types.
   - Elementwise ops enforce shape equality (dimension mismatch throws).
What is now true:

- Matrix-matrix matmul follows NxM rules (`(m,k) @ (k,n) -> (m,n)`) for supported dense numeric types and bit-packed dense bool.
- Elementwise ops enforce exact shape equality.
- Vector `@` rules are defined and exercised in tests:
   - `matrix @ vector` -> 1D vector
   - `vector @ matrix` -> row-shaped vector view (`(1,n)`)
   - `vector @ vector` -> scalar dot
- Python fallback matmul dispatch was updated to prefer native implementations and no longer assumes square outputs.
- The NxM-enabled vs restricted list is published as a user-facing page (see `guides/NxM Support.md`).

Acceptance checklist:

- Matmul follows NxM rules (`(N,M) @ (M,K) -> (N,K)`), and vector rules match NumPy 1D semantics.
- Elementwise ops enforce shape equality.
- Restricted/square-only ops throw deterministic errors on non-square inputs.
- The NxM-enabled vs restricted list is published in a discoverable docs location and kept current.

### Phase 3 — Structure policies for “inherently square” vs “shape-flexible”

**Status:** Complete (validated 2025-12-17)

We need explicit policies for:

- Triangular / causal matrices: square-only by definition.
- Symmetric / antisymmetric: square-only.
- Identity: shape-flexible (rectangular identity-like matrices are supported).
- Diagonal: square-only.

Decision captured: Diagonal matrices remain square-only in this milestone.

Acceptance checklist:

- All square-only structures validate `rows == cols` and raise deterministic exceptions otherwise.
- Documentation clearly states which structures are square-only.

Phase 3 completion record (2025-12-17):

- Square-only validation for inherently-square structures is enforced at creation/load/clone time (triangular/causal, diagonal, symmetric/antisymmetric).
- Identity matrices are explicitly shape-flexible (rectangular identity-like is supported).
- The square-only vs shape-flexible policy is documented in `documentation/dev/Square-only Assumptions.md`.

---

## 4) High-risk areas (things we must not get wrong)

1. **Accidental materialization / O(NM) zero-filling:**
   - `zeros`/`ones` necessarily imply a full write; the risk is performance and long I/O on out-of-core matrices.
   - Mitigation is explicit API: require `dtype`, provide `empty` for fast allocation, and provide `.fill(value)` to make initialization explicit.

2. **`size()` semantic flip ripple:**
   - Anything using `size()` as “dimension” will break.
   - We need a disciplined migration: use `rows()/cols()` for shape and keep `size()` only for total elements.

3. **Transpose views + CoW:**
   - With zero-copy transpose, `ensure_unique()` must be correct: mutating a transposed view must not mutate the base if shared.

4. **Persistence + transpose correctness:**
   - Must preserve base dims and flags; loading must reproduce the same logical behavior.


---

## 5) Acceptance checklist for Phase 1 (Float64 rectangular)

Phase 1 is done when all are true:

- Can create a dense Float64 matrix from a 2D rectangular NumPy array; shape preserved.
- Can create a dense Float64 matrix via `pycauset.zeros((rows, cols), dtype=float64)`.
- `get/set` works at edges for rectangular shapes.
- `T` produces a zero-copy view (no data duplication) with swapped logical shape.
- `pycauset.save` / `pycauset.load` roundtrip preserves shape and transpose.
- Square-only ops raise deterministic errors for non-square inputs.

---

## Appendix A) NxM support status (Phase 1)

Requirement: keep a simple, easily discoverable list of operations that are NxM-enabled vs NxM-disabled/restricted.

Initial intended status (to be validated against the actual exported API during implementation):

- Enabled in Phase 1: allocation, indexing, transpose views, NumPy import/export, persistence roundtrip, `.fill`.
- Restricted/square-only in Phase 1: inverse, determinant, eigen/symmetric eigen; and any structure whose definition is square-only (triangular, symmetric, antisymmetric, diagonal). Identity is shape-flexible.

This list should live in a place users will actually find (e.g. a dedicated doc page linked from the docs index), not only in an internals plan.

