# R1_LINALG — Linear Algebra Foundation (Release 1)

**Status:** Phase H completed (property-awareness sweep)

**Last updated:** 2025-12-21

> Documentation note:
>
> This file is a planning/spec artifact. User-visible R1 behavior is documented in:
>
> - [[guides/release1/linalg.md|Release 1: Linear Algebra (what shipped)]]
> - [[guides/Linear Algebra Operations.md|Linear Algebra Operations]] (workflow guide)
> - API reference for key endpoints (e.g., [[docs/functions/pycauset.solve.md|pycauset.solve]], [[docs/functions/pycauset.solve_triangular.md|pycauset.solve_triangular]])

## Purpose

Release 1 is not aiming to be a complete causal set library yet. The goal of **R1_LINALG** is to establish a **correct**, **well-specified**, and **architecture-compatible** linear algebra foundation that future work can optimize (CPU parallelism, GPU acceleration, and out-of-core streaming) without changing the public API.

In this milestone, it is acceptable to land some operations as **endpoints first**:

- a stable Python API function/method
- a single routing boundary through `ComputeContext` → `ComputeDevice` / `AutoSolver`
- CPU baseline implementation when feasible
- GPU/out-of-core paths can be added later behind the same entry points

## Key references

- Roadmap node: `documentation/internals/plans/TODO.md` → **R1_LINALG**
- Compute model: `documentation/internals/Compute Architecture.md`
- Shape/naming alignment: `documentation/project/protocols/NumPy Alignment Protocol.md`
- Op × dtype × device status: `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md`
- BLAS/LAPACK notes: `documentation/internals/plans/BLAS_INTEGRATION_PLAN.md`

## Relationship to R1_PROPERTIES (semantic properties)

R1 introduces a separate roadmap node **R1_PROPERTIES** that defines a canonical `properties` system.

This affects **R1_LINALG** in one important way:

- Some linalg operations are allowed (and expected) to **change behavior** when properties are set.

Therefore, even if the linalg surface is mostly complete today, **R1_LINALG is not considered fully complete until it has been audited for property-awareness after R1_PROPERTIES lands**.

This does *not* imply rewriting public endpoints. It is an implementation + documentation + tests alignment pass:

- Ensure operators that can exploit structure (`is_diagonal`, triangular properties, etc) do so.
- Ensure any operator whose behavior can change under properties documents that behavior.
- Ensure tests cover “same data, different properties” cases.

Correctness note:

- After R1_PROPERTIES, “correctness” for property-aware operators is defined **relative to the provided properties** (properties are gospel; we do not validate them).

## Constraints (non-negotiable)

- **Shapes:** only vectors (1D) and matrices (2D). No N-D tensors in R1.
- **One routing boundary per op:** frontend → `ComputeContext::get_device()` → `ComputeDevice`.
- **Correctness-first:** when an op is implemented, it must have a correct CPU baseline.
- **Safe routing:** if CUDA isn't implemented for an op, routing must be CPU-only (or a deterministic error).
- **Deterministic behavior:** supported shapes/dtypes/errors must be explicit and testable.
- **Docs + tests:** every public Python endpoint must have docs and tests.

## Phases

### Phase A  Precision mode contract & hook (COMPLETED)

Define and expose user-visible precision policy controls for *storage dtype selection*.

### Phase B  Inventory + contracts (COMPLETED)

Define the operation inventory and its contracts (shapes, dtypes, errors, routing), and reconcile docs against runtime.

### Phase C  Routing skeleton (COMPLETED)

Ensure operations route exclusively via `ComputeDevice`/`AutoSolver` so later CPU/GPU/out-of-core optimizations can slot in.

### Phase D  CPU baseline implementations (COMPLETED)

Implement correctness-first CPU versions for core operations introduced in this milestone.

### Phase E  Python surface + docs correctness (COMPLETED)

Deliverables:

- No phantom docs: everything documented must exist at runtime.
- Typed NumPy constructors and dtype/rank requirements are documented accurately.
- Reference pages follow Documentation Protocol.

### Phase F  SRP support-readiness table (COMPLETED)

Keep `SUPPORT_READINESS_FRAMEWORK.md` current (per-op CPU/GPU/out-of-core readiness) to avoid future archaeology.

### Phase G  Expand the linear algebra suite (COMPLETED — endpoint-first baseline)

Goal: define the R1 foundation endpoints for full linalg workflows (solve/factorize/spectral), even if optimized implementations arrive later.

Candidate endpoint families:

- Solves: `solve`, `lstsq`, `solve_triangular`
- Factorizations: `lu`, `cholesky` (and `ldlt` if needed)
- Spectral: `eigh/eigvalsh`, `eig/eigvals`
- SVD & pseudo-inverse: `svd`, `pinv`
- Stability utilities: `slogdet`, `cond`, `rank`

Each endpoint must:

- have a stable Python signature
- have a compute-backend hook (routing boundary)
- either have a CPU baseline implementation, or raise a deterministic “not implemented yet” error until implemented

### Phase H  Property-awareness alignment sweep (COMPLETED)

Completed items:

- Audit of property-aware routing across linalg endpoints: `solve` now short-circuits `is_identity`, rejects `is_zero`, and routes diagonal/triangular claims to `solve_triangular`; `matmul`/`solve_triangular`/`eigvalsh` already property-aware.
- Tests added for property-driven solve behavior (identity shortcut and zero singular guard).
- Docs updated for `solve` and `solve_triangular` to describe property-sensitive behavior.

### Phase I  Indexing & slicing (PENDING — numpy semantics for vectors/matrices)

Goal: implement NumPy-compatible slicing for 2D-only (matrices, and vectors represented as 1×N matrices) without introducing N-D tensors.

Scope (matches NumPy for rank-2):
- Basic indexing (`:`, integers incl. negative, slices with start/stop/step including negative, ellipsis, newaxis/None) yielding views where NumPy would (basic indexing → view) and copies where NumPy would (advanced indexing/Boolean/integer arrays → copy). Mixed basic+advanced follows NumPy’s copy semantics.
- Advanced indexing: integer arrays, boolean masks, mixed basic+advanced—copy semantics and NumPy shape rules.
- Dimensionality: mirrors NumPy reduction behavior (e.g., `M[i, :]` yields 1D in API, represented internally as 1×N matrix per existing convention). No shape-changing assignment beyond what NumPy allows for the indexed region.
- Assignment: `M[slice] = X` allowed; `X` must broadcast and convert per NumPy rules; dtype conversions emit PyCauset warnings per Warnings & Exceptions (e.g., promotion/overflow-risk); shape-changing assignments are rejected.
- Out-of-bounds and empty slices: NumPy rules (negatives wrap; empty slices are allowed and produce empty views/copies per NumPy behavior).

Storage/backing rules:
- Basic-indexing views must share backing without densifying and preserve device/storage. If a view cannot be represented without copy for a given structured type, raise a deterministic error (no silent densify or device hop).
- Persistent reuse: if the source is already persisted, large slices must reuse the existing on-disk backing as a persistent view rather than copy. If the source is only in-memory and the slice would exceed practical RAM, raise a deterministic error (no implicit spill/snapshot); a future opt-in spill policy can be added explicitly.

Deliverables:
- Contract doc updates (public API + internals) capturing the above semantics and warning triggers.
- Implementation of slicing/indexing paths honoring NumPy semantics within 2D-only constraint, with view/copy behavior matching NumPy.
- Tests covering basic/advanced indexing, assignment/broadcasting, dtype conversion warnings, OOB/empty slices, negative steps, and persistence-backed large slices.
- Documentation: reference docs must spell out view vs copy rules, persistence reuse/error behavior, assignment/broadcast semantics, warning categories used, and the 2D-only constraint.

Implementation contract (Phase I details):
- Allowed forms: `:`, integers (±), slices with step (±), ellipsis, `None`/newaxis, integer arrays, boolean masks, mixed basic+advanced per NumPy.
- View vs copy: basic → view; advanced or mixed → copy; no silent densify; preserve device/backing for views.
- Dimensionality: follow NumPy result shapes; vector user-facing results remain representable internally as 1×N matrices to stay 2D-only in storage.
- Assignment: allowed when the indexed region shape matches NumPy’s broadcast rules; dtype conversion emits `PyCausetDTypeWarning` (and `PyCausetOverflowRiskWarning` when applicable); shape-changing assigns raise.
- Large slices: reuse persisted backing when present; if only in-memory and too large for RAM, raise a deterministic error (no implicit spill). Deterministic errors when a structured type cannot form the view without copying.

Testing checklist (Phase I):
- Basic slices/views: positive/negative indices and steps, ellipsis, `None`, empty slices; verify view semantics (mutations reflect).
- Advanced indexing copies: integer arrays, boolean masks, mixed basic+advanced; verify copy semantics and NumPy shape parity.
- Assignment: broadcast success/failure cases, dtype-conversion warnings, overflow-risk warnings, rejection of shape-changing assigns.
- OOB/empty behavior: negatives wrap; empty slices produce empty results without error.
- Persistence/backing: slicing persisted matrices reuses backing; oversized slice on in-memory matrices raises deterministic error; device/backing preserved for views.
- 2D constraint: no accidental promotion to N-D; vectors remain 1×N internally.

#### Implementation status (as of 2025-12-21)
- Basic indexing (`:`, integers, slices, ellipsis) on dense matrices returns storage-sharing views when step==1; transposition/offset metadata is preserved. Structured types still reject slicing.
- Advanced indexing (1D integer arrays with negative wrap; 1D boolean masks) is supported per-axis with copy semantics; mixed basic+advanced also copies. Two array axes broadcast length or length-1; otherwise raise.
- Assignment (`__setitem__`) supports scalar, numpy 0/1/2-D arrays, or DenseMatrix RHS with NumPy-style broadcasting over the indexed region. Dtype casts trigger `PyCausetDTypeWarning`; narrowing/float→int casts also trigger `PyCausetOverflowRiskWarning`.
- Views with nonzero offsets are rejected by matmul/qr/lu/inverse kernels (require copy materialization first) to avoid incorrect strides.
- Not yet implemented: `None`/newaxis handling, persistence/backing policy (reuse persisted storage vs deterministic error for oversized in-RAM slices), and documentation/tests for persistence behavior.

## Notes

- Block matrices are tracked separately as **R1_BLOCKMATRIX** and should not be re-absorbed into this plan.
