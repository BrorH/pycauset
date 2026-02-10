# R1_CPU_PLAN — Modern Tiled CPU Engine (Phased Plan)

**Status**: Active (execution plan; work starts now)
**Owner**: Chief Programmer/Planner (AI)
**Stakeholder**: Chief Design Engineer (user)
**Last updated**: 2026-02-08

---

## 0) Scope and intent

R1_CPU makes the CPU a **first-class worker** in the same streaming architecture that powers GPU drivers. We are executing this plan now to modernize CPU kernels, integrate them with streaming drivers, and reach **≥ 0.90× NumPy** performance in the targeted regimes, without breaking the scale-first, out-of-core guarantees.

**Core Philosophy:**
1.  **Memory is limited, Disk is infinite.** We never throw OOM for purely large data; we must spill and compute out-of-core.
2.  **Hyper-optimized algorithms on disk.** Spilling to disk does not excuse slow $O(N^3)$ scalar loops. We must use blocked/tiled algorithms (like Recursive Blocked Inverse or Arnoldi) that operate efficiently on disk-backed objects.


---

## 1) Goals (what we must achieve)

- CPU kernels are **modern, tiled, and vectorized**, not legacy scalar loops.
- CPU can run **the same streaming drivers** as GPU through a **shared worker interface**, with no algorithm duplication.
- **Property-aware routing** remains consistent with `MatrixTraits`.
- **Streaming Manager** plans are respected (tile shapes, queue depth, routing reasons).
- **Performance** meets or exceeds **0.90× NumPy** for the defined regimes.
- **No regressions** in out-of-core behavior, dtype policies, or persistence semantics.
- Provide a **predictable, low-friction** path for adding **optimized future ops** that plugs into routing, streaming, metadata, and tests/docs.
- Block matrices are **explicitly supported** (or explicitly blocked) by the Op Contract, and priority ops account for block‑matrix semantics.

## 1.1) Documentation is mandatory (not optional)

Every phase **must** include a doc impact assessment and corresponding updates per `documentation/project/protocols/Documentation Protocol.md`.
The plan file itself is **not** documentation.
If any user-visible tuning knobs are introduced, they **must** be documented in API reference + guides + internals, with cross-links.

## 1.2) Advanced-user tuning knobs (allowed, explicit)

R1_CPU may introduce **advanced-user** tuning knobs for performance and routing, provided they are:

- explicit (opt-in),
- minimal and stable,
- tested, and
- fully documented as **advanced controls**.

Candidate knobs (approved in principle):

- Streaming threshold controls
- Tile size override (advanced)
- Thread count override
- CPU backend preference (ParallelFor vs OpenMP)
- Trace verbosity / routing diagnostics

---

## 2) Non-goals (out of scope)

- Unrelated new public APIs or user-visible features.
- Performance tuning/diagnostic controls **are allowed** when necessary, but must be explicit, minimal, opt-in, tested, and documented per protocol.
- New storage formats or metadata schemas.
- CUDA kernel work (covered by R1_GPU).
- Any change that weakens export guards or allows silent huge materialization.

---

## 3) Architectural intent (summary)

### 3.1 Shared worker interface (naming + semantics)
The GPU does **not** use a CPU tool. The intent is a **shared worker interface** that both CPU and GPU implement. The neutral name is **`ComputeWorker`**.

**Rule**: Streaming drivers call the **worker interface**, never CPU- or CUDA-specific APIs directly.

### 3.2 Streaming-first CPU
CPU execution should respect the Streaming Manager’s plan. The plan controls:
- `route` (`direct` vs `streaming`)
- `tile_shape` (clamped to shapes)
- `queue_depth` (bounded)
- `trace_tag` and events

### 3.3 Direct-path guardrails
When `MemoryGovernor::should_use_direct_path(...)` is true, CPU should follow the **direct** path (no tiling) to avoid “nanny” overhead.

---

## 4) Dependencies and references (read-only)

- `documentation/dev/index.md`
- `documentation/dev/Codebase Structure.md`
- `documentation/dev/Build System.md`
- `documentation/dev/Python Internals.md`
- `documentation/dev/Bindings & Dispatch.md`
- `documentation/dev/Warnings & Exceptions.md`
- `documentation/dev/Testing & Benchmarks.md`
- `documentation/dev/Repository Hygiene.md`
- `documentation/dev/Restructure Plan.md`
- `documentation/dev/Square-only Assumptions.md`
- `documentation/internals/index.md`
- `documentation/internals/Compute Architecture.md`
- `documentation/internals/Streaming Manager.md`
- `documentation/internals/MemoryArchitecture.md`
- `documentation/internals/Memory and Data.md`
- `documentation/internals/CooperativeArchitecture.md`
- `documentation/internals/Algorithms.md`
- `documentation/internals/DType System.md`
- `documentation/project/Philosophy.md`
- `documentation/project/protocols/*`

---

## 5) Decisions locked (policy)

1) **Worker interface name**: `ComputeWorker` (shared CPU/GPU worker interface).
2) **Backend selection policy**: **build-time only** (no runtime auto-probing; advanced override only if needed).
3) **Threading backend policy**: benchmark **ParallelFor/thread pool** vs **OpenMP** and pick the winner; prefer the more established option when performance is comparable.
4) **SIMD policy**: scalar baseline, runtime dispatch for **AVX2**, optional **AVX‑512** when available.
5) **Tile policy**: default to Streaming Manager tile sizes; CPU overrides only if documented and tested.

---

# Execution plan (phased)

Each phase below is work we will do now. Every phase includes **implementation**, **testing**, and **documentation** tasks. No phase is complete without all three.

---

## Phase 1 — Contract lock + audit (COMPLETE)

**Objective**: Establish the CPU worker contract and audit existing CPU kernels/legacy loops (start immediately).

### Implementation tasks
- [x] Define the **worker interface contract** used by streaming drivers (method signatures, required behaviors, error reporting).
- [x] Adopt and document **`ComputeWorker`** as the shared worker interface name.
- [x] Clarify architecture: **`CpuSolver`** remains the concrete CPU algorithm engine; **`ComputeWorker`** is the abstract streaming interface.
- [x] Implement a CPU backend for `ComputeWorker` (e.g., `CpuStreamWorker`) that delegates to `CpuSolver`.
  - Rationale: The worker interface is **shared by CPU and GPU** streaming drivers. `CpuSolver` contains the low-level math kernels.
- [x] Audit current CPU kernels to identify **legacy loops** and **tile‑unsafe code paths**.
- [x] Identify the **priority op list** (matmul, elementwise, inverse, eigh/eigvalsh) and map current implementations to target replacements.
- [x] Audit **block matrix** pathways for priority ops and identify explicit support/unsupported cases.
- [x] Inventory existing **user-facing performance controls** (thresholds, routing overrides, budgets) and decide if any **new tuning knobs** are required for R1_CPU.
- [x] Define the **advanced-user tuning knobs** set (names, defaults, scope, safety constraints).

### Testing tasks
- [x] Add a **contract test** that validates streaming drivers can call a dummy CPU worker implementation (no GPU required).
- [x] Add a **routing test** to ensure streaming plans are recorded in IO trace when forced by threshold.

### Documentation tasks
- [x] Update **internals** to describe the worker interface and streaming driver contract:
  - `documentation/internals/Compute Architecture.md`
  - `documentation/internals/Streaming Manager.md`
  - `documentation/project/protocols/Adding Operations.md`
- [x] Add a short **dev handbook** note for contributors describing where CPU worker lives and how to extend it.
- [x] Complete the **Documentation Protocol** impact assessment for this phase and record which pages were updated.
- [x] If any tuning knobs are introduced, add **advanced-user** guidance in:
  - API reference (parameters/behavior)
  - Guides (with explicit “advanced” framing)
  - Internals (routing/implementation details)

### Exit criteria
- [x] Worker interface name + contract is decided and documented.
- [x] Audit report exists (list of legacy loops and target replacements).
- [x] Contract tests pass.

---

## Phase 2 — Op Contract + Registry scaffolding (pre‑ops) (COMPLETE)

**Objective**: Establish the Op Contract and registry so all priority ops implement through it from day one.

### Implementation tasks
- [x] Define a **CPU Op Contract** (C++ authoritative + Python mirror) that captures:
  - shape/dtype/structure support
  - **block‑matrix support**: status (supported / leaf‑only / blocked), block layout assumptions, and routing rules
  - **block‑aware execution rules**: whether the op can exploit block structure, and how block access/stride affects tiling/streaming
  - streamability (optional) and access pattern
  - property consumption/propagation rules (no scans)
  - lazy‑metadata rules (scalar/transpose/conjugation handling)
  - compute‑worker hooks (tile compute signatures, scratch needs)
- [x] Implement a lightweight **Op Registry** that maps op → contract + hooks (authoritative in C++), with a **Python mirror** for docs/tests/AI guidance.
- [x] Add a **ComputeWorker extension template** so new ops only implement tile kernels + contract registration.
- [x] Define **optimization tiers** (Tier‑1/Tier‑2/Tier‑3) with clear expectations for SIMD, streaming, and property‑aware shortcuts.

### Testing tasks
- [x] Contract tests that validate every op declares streamability, property usage, and lazy‑metadata rules in one place.
- [x] Contract tests that validate **block‑matrix support status** is declared and enforced.

### Documentation tasks
- [x] Add an **internals** page section describing the Op Contract + Op Registry and how to extend it.
- [x] Add **dev handbook** guidance for contributors/AI agents: “how to add an optimized op.”

### Exit criteria
- [x] Op Contract + Registry exist in C++ with Python mirror.
- [x] ComputeWorker extension template is in place.
- [x] Tier expectations are documented.

---

## Phase 3 — CPU matmul via shared worker + tiling (COMPLETE)

**Objective**: Replace legacy CPU matmul with a tiled worker path consistent with streaming drivers **and the Op Contract**.

### Implementation tasks
- [x] Implement CPU worker tile compute for matmul (blocked, cache‑aware) using the contract/registry path.
- [x] **Audit Action**: Deprecate/Remove the legacy `CpuSolver::matmul_streaming` (which creates its own unchecked plan).
- [x] **Audit Action**: Implement a public `gemm(alpha, beta)` API in `CpuSolver` to allow `CpuComputeWorker` to accumulate tiles without allocating temporary matrices or re-running type dispatch.
- [x] Integrate with streaming driver path so **matmul** uses the shared worker for CPU execution when streaming is chosen.
- [x] Respect MemoryGovernor’s **direct path** (pinning/anti‑nanny) before falling back to streaming.
- [x] Ensure MatrixTraits/property‑aware routing is honored.
- [x] Ensure **block‑matrix matmul** routes through the Op Contract and decomposes into leaf operations without densification.

### Testing tasks
- [x] Correctness tests:
  - Dense float32/float64 matmul vs NumPy.
  - Rectangular shapes (NxM × MxK) with multiple tile sizes.
- [x] Block‑matrix matmul tests (mixed dtypes/structures at leaf level) with explicit routing assertions.
- [x] Routing tests:
  - Force streaming via threshold and assert IO trace includes streaming plan.
  - Direct path chosen when `allow_huge=True` or data fits in RAM.
- [x] Performance smoke test:
  - Compare to NumPy on a small suite; fail only if severe regression (warn threshold).

### Documentation tasks
- [x] Update **internals** with CPU matmul path and streaming integration details:
  - `documentation/internals/Compute Architecture.md`
  - `documentation/internals/Streaming Manager.md`
- [x] Add a short **performance guide** note if matmul routing behavior changes.
- [x] If a new tuning knob is introduced in this phase, update **API reference** + **guide** pages and add cross-links.

### Exit criteria
- [x] Legacy matmul loop is removed or gated off.
- [x] Matmul passes correctness tests and routing tests.
- [x] Performance baseline recorded.

---

## Phase 4 — Vectorized elementwise ops (COMPLETE)

**Objective**: Modernize elementwise CPU ops with SIMD and shared worker path.

##**Audit Action**: Replace legacy scalar `ParallelFor` loops for integers (Int8/16/32/64) with SIMD-aware kernels (using `xsimd` or intrinsics) to match float performance.
- [x] Implement CPU worker tile compute for elementwise add/sub/mul/div and scalar ops.
- [x] Add runtime SIMD dispatch (scalar → AVX2 → AVX‑512).
- [x] Ensure dtype rules (anti‑promotion, overflow behavior) remain unchanged.
- [x] Remove legacy scalar `ParallelFor` loops for integers (Int8/16/32/64) with SIMD-aware kernels.

### Testing tasks
- [x] Correctness tests for all elementwise ops across float32/float64/int types.
- [x] Vectorization safety tests: verify results match scalar path.
- [x] Routing tests to ensure streaming plan honored for large data.

### Documentation tasks
- [x] Update **internals** to describe SIMD policy and CPU elementwise path:
  - `documentation/internals/Compute Architecture.md`
- [x] Update relevant **API references** if any user-visible behavior changes (should be none).
- [x] If a new tuning knob is introduced in this phase, update **API reference** + **guide** pages and add cross-links.

### Exit criteria
- [x] Legacy elementwise loops removed or disabled.
- [x] SIMD paths validated against scalar results.

---

## Phase 5 — Inverse (BLAS/LAPACK‑first) (COMPLETE)

**Objective**: Implement inverse through BLAS/LAPACK paths, using the Op Contract + ComputeWorker, and replace legacy CPU inverse code paths.

### Implementation tasks
- [x] **Audit Action**: Remove the legacy manual block Gauss-Jordan implementation.
- [x] Route inverse through BLAS/LAPACK implementations where available; avoid custom kernels unless required.
- [x] Ensure non‑square guards remain deterministic and route to direct with a clear reason.
- [x] Respect property‑aware fast paths (e.g., SPD → Cholesky) when properties allow short‑circuiting.
- [x] Declare and enforce **block‑matrix support status** for inverse (supported or explicitly blocked with reason).
- [x] Replace legacy CPU inverse code paths with the new contract‑based implementation.

### Testing tasks
- [x] Correctness tests vs NumPy/Scipy for small/medium sizes.
- [x] Regression tests for non‑square behavior.
- [x] Block‑matrix tests for declared support or deterministic rejection (with reason).

### Documentation tasks
- [x] Update **internals** for CPU inverse path coverage and routing reasons:
  - `documentation/internals/Compute Architecture.md`
  - `documentation/internals/Streaming Manager.md`
- [x] Add/update any **developer debugging notes** (trace tags, routing explanations).
- [x] If a new tuning knob is introduced in this phase, update **API reference** + **guide** pages and add cross-links.

### Exit criteria
- [x] Inverse uses the Op Contract + ComputeWorker path.
- [x] Guards and routing reasons are deterministic and tested.

---

## Phase 6 — Eigen (Hermitian + non‑Hermitian + Arnoldi)

**Objective**: Implement Hermitian and non‑Hermitian eigen solvers using LAPACK where available, plus Arnoldi for large/top‑k, all via the Op Contract + ComputeWorker.

##**Audit Action**: Wire LAPACK `dsyev` for `eigh` (currently missing/fallback-only in `CpuSolver`).
- # Implementation tasks
- Use LAPACK‑backed paths for `eigh`/`eigvalsh` when available; avoid custom kernels until necessary.
- Add **non‑Hermitian eigen solver** support (`eig`/`eigvals`) with LAPACK‑backed paths when available.
- Implement and optimize **Arnoldi** (or update/replace any existing implementation) so it uses the Op Contract + ComputeWorker path.
- Declare and enforce **block‑matrix support status** for eigen ops (supported or explicitly blocked with reason).
- Replace legacy CPU eigen code paths with the new contract‑based implementation.
- Persist eigenvalues and eigenvectors as **cached‑derived metadata** (big‑blob caches when large), using stable signatures and no implicit recompute on cache miss.

### Testing tasks
- Correctness tests vs NumPy/Scipy for small/medium sizes with allowed variance up to **1%**.
- Streaming route tests under forced thresholds.
- Regression tests for non‑square behavior.
- Block‑matrix tests for declared support or deterministic rejection (with reason).
- Arnoldi correctness tests (including top‑k eigenvalues) and routing/streaming assertions when applicable.
- Cache tests for eigen metadata: load‑hit, load‑miss (warning + ignore), and signature‑mismatch behavior.
- Non‑Hermitian eigen correctness tests (eigenvalues and eigenvectors) with routing/streaming assertions when applicable.

### Documentation tasks
- Update **internals** for CPU eigen path coverage and routing reasons:
  - `documentation/internals/Compute Architecture.md`
  - `documentation/internals/Streaming Manager.md`
- Add/update any **developer debugging notes** (trace tags, routing explanations).
- If a new tuning knob is introduced in this phase, update **API reference** + **guide** pages and add cross-links.
- Document Arnoldi support and its routing/constraints in internals and relevant API reference pages.
- Document eigenvalue/eigenvector caching semantics (big‑blob caches, signature validation, and warning behavior on miss).
- Document non‑Hermitian eigen support (`eig`/`eigvals`) and its routing/constraints.

### Exit criteria
- Hermitian and non‑Hermitian eigen ops use the Op Contract + ComputeWorker path.
- Guards and routing reasons are deterministic and tested.

---

## Phase 7 — Future optimized ops (extensible CPU pipeline)

**Objective**: Make it straightforward to add optimized CPU ops beyond the priority trio, without scattering policy or violating metadata/streaming rules.

### Implementation tasks
- Add initial optimized implementations for the following ops (Tier‑2 unless noted):
  - determinant (LU `getrf` + diagonal product; SPD → `potrf`)
  - trace (direct diagonal sum; streaming‑safe)
  - norm (1/∞/Frobenius only; spectral norm out of scope for R1_CPU)
  - solve (`getrf/getrs` general; `potrf/potrs` SPD)
  - svd (`gesdd` default, fallback to `gesvd`)
  - qr (`geqrf` + `orgqr/ungqr`; pivoted QR `geqp3` when needed)
  - lu (`getrf`)
  - cholesky (`potrf`)
  - rank (pivoted QR `geqp3`, fastest path)
- Cache heavy factorizations/results (LU/QR/SVD) as **cached‑derived metadata** where appropriate, using big‑blob caches when large and no implicit recompute on cache miss.

### Testing tasks
- Correctness tests for each new op across supported dtypes/structures.
- Routing tests for streaming vs direct (where applicable).
- Performance smoke tests for Tier‑2 ops vs NumPy (targets scoped and documented).
- Cache tests for LU/QR/SVD metadata: load‑hit, load‑miss (warning + ignore), and signature‑mismatch behavior.

### Documentation tasks
- Update **API reference** + **guides** for each new op (explicitly label any advanced knobs).
- Ensure Documentation Protocol checklist is satisfied for each new op.

### Exit criteria
- At least one Tier‑2 op is fully wired through contract → worker → tests → docs as a reference implementation.
- All listed ops have explicit support status (implemented or deliberately blocked with clear reasons).

---

## Phase 8 — Cleanup, parity gates, and SRP alignment

**Objective**: Remove remaining legacy CPU paths, enforce ≥ 0.90× NumPy target, and align with SRP.

### Implementation tasks
- Remove or hard‑gate all legacy loops not used by the new worker path.
- Align AutoSolver routing with MatrixTraits for all CPU‑covered ops.
- Ensure consistent trace annotations across CPU/GPU/streaming paths.

### Testing tasks
- SRP‑aligned coverage grid for CPU: dtype × op × structure × storage.
- Performance gates for priority ops against NumPy (>0.90×).
- Out‑of‑core stress test (streaming + mmap) for at least one large op.

### Documentation tasks
- Update **internals** with final CPU architecture details and extension guidance.
- Update **dev handbook** to reflect new CPU execution pathway and tests.
- Ensure docs meet the Documentation Protocol checklist.
- If any new tuning knobs were added, ensure **API reference** + **guides** + **internals** are all updated and cross-linked.

### Exit criteria
- All targeted ops meet performance threshold.
- Legacy CPU loops removed.
- Docs and tests pass in SRP‑aligned gates.

---

## 6) Risks and mitigations

- **Risk**: CPU overrides tile policy diverge from Streaming Manager.
  - **Mitigation**: Default to Streaming Manager; overrides must be documented and tested.
- **Risk**: SIMD path correctness drift.
  - **Mitigation**: Scalar reference tests + randomized property tests.
- **Risk**: Hidden materialization in streaming path.
  - **Mitigation**: Add routing tests that assert no implicit export/materialization.

---

## 7) Definition of Done (R1_CPU)

- CPU kernels are tiled, vectorized, and integrated via shared worker interface.
- Streaming drivers are reusable across CPU/GPU with deterministic routing.
- Performance: ≥ 0.90× NumPy for priority ops in defined regimes.
- Tests + documentation updated per protocol in **every phase**.

---

## 8) Start now (next actions)

- Finalize the benchmark plan for **ParallelFor vs OpenMP** selection.
- Confirm the **advanced-user tuning knobs** list and documentation targets.
- Begin Phase 1 execution.
