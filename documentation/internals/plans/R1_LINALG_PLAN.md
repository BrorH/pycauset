# R1_LINALG Plan — Core Linear Algebra Surface Expansion (Correctness-First)

**Status:** Active (drafted 2025-12-17; progress tracking enabled)

## Progress snapshot (MUST keep updated)

This section is **non-optional**. For the rest of this milestone, every time work is done, this snapshot must be updated so it is always obvious:

- where we are in the phase steps,
- what is done,
- what is next.

**Last updated:** 2025-12-17

**Current phase step:** Phase C + Phase D (subset) — Minimal norm(x) support

**What was just completed (DONE):**

- Implemented NumPy-style 2D broadcasting for elementwise matrix ops (`+`, `-`, `*`) in the CPU solver.
- Updated Python bindings so elementwise ops accept NumPy 1D/2D arrays.
- Added Python tests for NumPy 1D broadcasting cases.
- Added docs note explaining elementwise broadcasting rules.
- Added elementwise division (`/`) for matrices, including NumPy 1D/2D operands.
- Added `pycauset.divide(a, b)` convenience wrapper + docs.
- Implemented minimal `pycauset.norm(x)` support end-to-end:
   - vector $\ell_2$ norm (via `ComputeDevice::l2_norm`),
   - matrix Frobenius norm (via `ComputeDevice::frobenius_norm`),
   - CPU implementations with AutoSolver CPU routing,
   - Python binding + facade wrapper,
   - docs + tests.

**What is next (NEXT):**

- Precision mode contract & hook (explicit precision/underpromotion control surface).
- Block matrices have been promoted to a dedicated roadmap node: **R1_BLOCKMATRIX**.
   See `documentation/internals/plans/R1_BLOCKMATRIX_PLAN.md`.

**What is blocked / deferred:**

- Precision control surface (explicit `PrecisionMode` / config) is not designed yet.
- GPU broadcasting kernels are not implemented yet (CPU fallback is used where applicable).

## 0) Purpose and scope

R1_LINALG expands PyCauset’s linear algebra surface area while keeping the system architecture scalable:

- **Correctness first:** new ops must be correct and deterministic across supported shapes/dtypes/structures.
- **Architecture-respecting:** new ops must route through the compute abstraction layer (AutoSolver / ComputeDevice), so later R1_GPU and R1_IO improvements can land behind stable entry points. See TODO.md
- **No scope creep:** no N-D tensors; only vectors (1D) and matrices (2D).
- **No “NumPy clone” mandate:** we keep familiar naming where it helps, but we do not attempt full `numpy.linalg` parity.

Key references:

- Roadmap node: `documentation/project/TODO.md` → **R1_LINALG**.
- Compute routing & device model: `documentation/internals/Compute Architecture.md`.
- Naming/shape expectations: `documentation/project/protocols/NumPy Alignment Protocol.md`.
- Dtypes × ops × devices gates: `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md`.
- CPU BLAS backend notes: `documentation/internals/plans/BLAS_INTEGRATION_PLAN.md`.

Non-goals (explicit):

- Implementing or optimizing CUDA kernels for every new op (belongs to R1_GPU/SRP).
- Implementing out-of-core optimized streaming for every new op (belongs to R1_IO/SRP).
- Broadcast semantics beyond what is already explicitly supported.

---

## 1) Guiding policies (requirements)

### 1.1 Public API policy

- Keep top-level naming consistent with existing `pycauset` conventions (snake_case, lower-case factories).
- Do not introduce a parallel `pycauset.linalg.*` namespace in this milestone.
- Each new public op must have a reference doc page and tests.

### 1.2 Shape rules

- Vectors are 1D with `shape == (n,)`.
- Matrices are 2D with `shape == (rows, cols)`.
- Elementwise-style ops use **NumPy-style 2D broadcasting**.
- Matmul/matvec rules follow the established Phase 2 rules.

NumPy alignment note (explicit decision):

- `matmul(a, b)` on incompatible shapes raises a deterministic error (no implicit broadcasting).

Elementwise broadcasting note (explicit decision):

- Elementwise ops (`+`, `-`, `*`, and future `divide`) broadcast like NumPy for **2D** operands.
- When mixing a matrix with a **1D NumPy array** in elementwise ops, the array is treated as a **row vector** with shape `(1, n)`.

### 1.3 Dtype + promotion (“underpromotion mantra”)

Decision (explicit):

- PyCauset strongly prefers **downward promotion** when it improves speed and memory footprint.
- In particular, GPU execution may intentionally compute using float32 even when inputs originate as float64.

Requirements (to avoid “surprise”):

- The precision policy must be **explicit and documented**.
- The behavior must be **consistent and testable** (no “sometimes it downcasts” ambiguity).
- Provide an escape hatch to force higher precision when needed (likely a `PrecisionMode` / config setting). The exact configuration surface can be introduced in R1_GPU, but R1_LINALG ops must route through the compute boundary so the policy is centralized.

This plan intentionally separates:

- **storage dtype** (what the object is), vs
- **compute dtype** (what an accelerator might use internally).

### 1.4 Compute Architecture integration

### 1.4 Compute Architecture integration

All new ops must be expressed as:

1) a stable public entry point (Python and/or C++ frontend), then
2) a single routing boundary that calls AutoSolver/ComputeDevice, then
3) a CPU implementation as baseline correctness,
4) optional GPU implementation and optional streaming implementation later.

I/O hinting (prefetch/discard) should happen at the solver boundary, not sprinkled into leaf kernels.

### 1.5 External solver / dependency policy (don’t reinvent the wheel)

PyCauset should prefer established, well-tested numeric libraries for heavyweight linear algebra where appropriate.

Guidelines:

- Use external libraries when they reduce complexity and improve correctness/performance (examples: BLAS/LAPACK for GEMM and decompositions; cuBLAS/cuSOLVER on CUDA).
- Do not re-implement complex solvers (e.g., eigensolvers) unless there is a clear project-specific reason.
- Keep a **single dispatch boundary** per operation that routes via AutoSolver/ComputeDevice. External calls happen behind that boundary (CpuSolver/CudaSolver implementations), not at random call sites.
- Out-of-core streaming remains PyCauset’s “chassis”: external libs are the “engine” operating on tiles/buffers.

---

## 2) Work breakdown

### Milestone 0 — Elementwise broadcasting alignment (DONE 2025-12-17)

Goal: align elementwise matrix ops with NumPy-style broadcasting while preserving the compute routing boundary.

Deliverables (completed):

1. CPU baseline for broadcasted elementwise ops.
2. Python bindings support for NumPy 1D/2D operands.
3. Tests covering broadcast success + mismatch failure.
4. Docs describing broadcasting behavior.

### Milestone 1 — Elementwise division (DONE 2025-12-17)

Goal: add elementwise division with the same broadcasting rules as other elementwise ops.

Deliverables (completed):

1. Promotion contract for division (int/uint/bit defaults to float64 when neither operand is float/complex).
2. CPU baseline kernel via the existing broadcast-aware binary-op machinery.
3. Python bindings for `/` with NumPy 1D/2D operands.
4. `pycauset.divide(a, b)` convenience helper + docs + tests.

### Note — Block matrices moved out of R1_LINALG

Block matrices turned into a large enough effort (heterogeneous dtypes, manifests, views,
semi-lazy evaluation) that they are tracked as a dedicated Release-1 node:

- Roadmap: `documentation/project/TODO.md` → **R1_BLOCKMATRIX**
- Canonical plan: `documentation/internals/plans/R1_BLOCKMATRIX_PLAN.md`

### Phase A — Inventory + contracts (SRP-0 lite)

Goal: agree on *what* exists before coding.

Deliverables:

1. Define the initial R1_LINALG operation inventory.
2. For each op, specify:
   - accepted shapes (vector vs matrix; square-only requirements),
   - dtype rules (what dtypes are supported; what happens on mixed dtypes),
   - structure restrictions (triangular/symmetric/diagonal/identity behavior),
   - deterministic error messages/classes for unsupported cases.
3. Create/extend a minimal “support table skeleton” (op × dtype × device × storage) that can be filled in during SRP.

Acceptance checklist:

- Every op has a written contract (shape + dtype + restriction + error behavior).
- Contracts reference the compute architecture boundaries.

### Phase B — Routing skeleton (ComputeDevice first)

Goal: make later GPU/IO work drop-in.

Deliverables:

1. Ensure each op has a single dispatch point that routes via AutoSolver/ComputeDevice.
2. Ensure CPU fallback is explicit and testable.
3. Ensure device selection policy is observable (even if simple today).

Acceptance checklist:

- Every op routes through a stable compute interface.
- No hidden “side paths” bypass the device model.
- Ops that use external libraries still route through the same device boundary.

### Phase C — CPU baseline implementations

Goal: correctness and determinism, with reasonable performance.

Implementation approach:

- Prefer reusing existing primitives (matmul, elementwise, reductions) to avoid duplicated kernels.
- Keep memory behavior predictable (avoid accidental materialization when not required).

Acceptance checklist:

- Unit tests cover correctness, edge shapes, dtype behavior, and failure cases.
- CPU implementation exists for each op in the initial inventory.

### Phase D — Python surface + docs

Goal: users can actually use the new operations.

Deliverables:

- Public Python entry points for each op.
- Reference docs pages + examples.

Acceptance checklist:

- Docs updated per `Documentation Protocol.md`.
- Examples are consistent with current naming.

### Phase E — SRP handoff hooks (not full SRP)

Goal: make SRP/GPU/IO work low-friction later.

Deliverables:

- For each op, record in the SRP framework:
  - CPU correctness status,
  - GPU route status (CPU-route vs GPU-enabled vs blocked),
  - out-of-core status (naive vs streaming-enabled).

---

## 3) Proposed initial operation inventory (editable)

This is a starting spine (not final). The point is to pick ops that are foundational and reused by others.

### 3.1 Reductions / norms

- `norm(x, ord=..., axis=..., keepdims=...)` (start minimal: vector 2-norm + matrix Frobenius; expand later)
- `sum(x, axis=None, keepdims=False)` (if not already present / if needed as a building block)

### 3.2 Elementwise division + scaling

- `divide(a, b)` elementwise (shape-equal). Should also work via "a / b".
- `divide_scalar(a, s)` / `scale(a, s)`

### 3.3 Initialization from array input for typed classes

- Allow `FloatMatrix(np_array)` / `FloatVector(np_array)` (and other typed classes) with clear dtype rules.

### 3.4 Block operations (phase-later within R1_LINALG)

- Block matrix construction / concatenation / slicing helpers (design needed; avoid memory bombs) THIS PART WILL REQUIRE SOME DISCUSSION BEFORE IS IMPLEMENTED! I HAVE MANY NOTES!

### 3.5 Advanced indexing (phase-later within R1_LINALG)

- Slicing / fancy indexing (likely large surface; design carefully)

### 3.6 Random generation (phase-later)

- `rand(shape, dtype=..., seed=...)` / `randn(...)` (if desired)

### 3.7 Matrix properties (phase-later)

- `is_symmetric`, `is_positive_definite`, etc. (must define tolerances and dtype behavior)

---

## 4) Risk register (what will cause rework later)

1. **Promotion/surprise precision changes**
   - Downward promotion is desired for speed/space, but results may differ from float64 compute.
   - Mitigation: make precision policy explicit, consistent, and overrideable.

2. **Broadcast semantics confusion**
   - Allowing implicit broadcasting can make ops ambiguous and harder to optimize.
   - Mitigation: keep strict shape equality unless explicitly allowed per op.

3. **Out-of-core performance cliffs**
   - Naive implementations can thrash disk.
   - Mitigation: keep I/O hinting at solver boundary; record “naive” vs “streaming” status for SRP.

4. **Surface bloat**
   - Too many ops too quickly makes SRP explode.
   - Mitigation: pick a small spine first; expand iteratively.

---

## 5) Open questions (must settle before implementation)

1. **Precision control surface:** what is the user-facing control for precision mode (e.g., `AUTO` prefers device precision; `FORCE_FLOAT64` disables downcast)?
2. **Initial spine:** confirm which 2–3 areas to implement next (division, norms, typed init-from-array, block ops, advanced indexing, random, properties).

---

## 6) Acceptance criteria for R1_LINALG (definition of done)

R1_LINALG is done when:

- The initial op inventory is implemented with CPU correctness and deterministic errors.
- Each op routes through the compute abstraction layer.
- Docs exist for each public op, with examples.
- Tests exist per op covering shape rules and dtype rules.
- A support-status skeleton exists (or is updated) so SRP/GPU/IO can systematically fill in performance/device/storage coverage later.
