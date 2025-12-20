# R1_LINALG — Linear Algebra Foundation (Release 1)

**Status:** Active

**Last updated:** 2025-12-20

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

## Relationship to R1_FLAGS (property flags)

R1 introduces a separate roadmap node **R1_FLAGS** that defines a canonical `flags` system.

This affects **R1_LINALG** in one important way:

- Some linalg operations are allowed (and expected) to **change behavior** when flags are set.

Therefore, even if the linalg surface is mostly complete today, **R1_LINALG is not considered fully complete until it has been audited for flag-awareness after R1_FLAGS lands**.

This does *not* imply rewriting public endpoints. It is an implementation + documentation + tests alignment pass:

- Ensure operators that can exploit structure (`is_diagonal`, triangular flags, etc) do so.
- Ensure any operator whose behavior can change under flags documents that behavior.
- Ensure tests cover “same data, different flags” cases.

Correctness note:

- After R1_FLAGS, “correctness” for flag-aware operators is defined **relative to the provided flags** (flags are gospel; we do not validate them).

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

### Phase H  Flag-awareness alignment sweep (REQUIRED after R1_FLAGS)

Goal: finalize the R1 linalg foundation under the canonical `flags` system.

Deliverables:

- Audit existing linalg endpoints and ensure that any operation that can exploit flags is explicitly flag-aware.
- The sweep must cover, at minimum, the canonical flag families introduced in `R1_FLAGS_PLAN.md`:
	- `is_zero` short-circuits
	- diagonal / triangular structure (including strict + unit-diagonal)
	- `is_identity`
	- `is_permutation`
	- symmetry family (`is_symmetric`, `is_anti_symmetric`, `is_hermitian`)
	- `is_unitary`
	- `is_atomic` (where relevant to any matrix-function endpoints)
- Add tests that demonstrate that flags can change operator behavior deterministically.
- Update reference docs for affected endpoints to state how flags influence computation.

## Notes

- Block matrices are tracked separately as **R1_BLOCKMATRIX** and should not be re-absorbed into this plan.
