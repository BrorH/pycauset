# R1_IO — Out-of-core I/O + Persistence Performance (Phased Plan)

**Status:** Draft (to be iterated)

**Goal:** Disk-backed operation performance is a first-class feature, not an accident.

This plan focuses on making large-scale workflows (10s–100s of GB) correct, efficient, and predictable without hidden full materialization.

## Scope and non-goals

### In scope

- Snapshot I/O (`save`/`load`) correctness and performance.
- Out-of-core compute behavior (direct-vs-streaming selection, paging friendliness).
- IO hinting (prefetch/discard) correctness and measurable impact.
- Explicit temp vs snapshot semantics (spill/eviction `.tmp` vs `.pycauset` snapshots).
- The **NumPy conversion surface**, as it intersects with I/O and out-of-core safety.

### Non-goals (R1)

- Building a new mmap/NumPy backend to make NumPy handle massive arrays “better than NumPy itself”.
- Zero-copy NumPy views for huge matrices if it requires substantial new storage architecture. (If this architecture is very stable and low-maintanence, however, I say it is very worth exploring)
- New *internal snapshot container* formats beyond the single `.pycauset` format.
  - R1 may still add **import/export** for other formats; it just must not fork the canonical on-disk snapshot format.

## Terms and invariants (must remain true)

- **Snapshot**: a `.pycauset` file created by `save()`; treated as immutable when loaded.
- **Working storage / spill**: switching an object to file-backed mapping via temporary session files (e.g., `.tmp`).
- **Payload vs metadata**:
  - payload bytes may live on disk and be paged in;
  - metadata/properties/cached-derived values are logically separate and must round-trip correctly on snapshot save/load.
- **Scale-first**: new features must not introduce accidental full scans, accidental densification, or hidden full copies for out-of-core operands.

## Working protocol (agent ↔ design chief)

- Operate phase-by-phase: state the current phase, list intended edits, then execute.
- Ask clarifying questions when uncertain; propose 1–3 options when choices exist.
- After edits, report “what changed” and “what’s next”; keep this loop for all future plan/doc work.
- Explicitly mark when a phase is complete once its Definition of Done is met.

## Phase 0 — Contract lock (interfaces + policies) — **Status: Completed**

**Objective:** Decide the policies that other phases implement and test.

Deliverables:

1. **Conversion surface (NumPy interop policy, I/O-aware)**
   - PyCauset does not expose a `pc.asarray` “array” API; only matrices/vectors are first-class.
   - Define supported conversion entrypoints and their semantics:
     - `pc.matrix(np_array)` / `pc.vector(np_array)` (import from NumPy)
     - `np.asarray(obj)` / `np.array(obj)` (export to NumPy)
   - **Export-to-NumPy policy (locked):**
     - **Huge / out-of-core objects:** `np.asarray(obj)` / `np.array(obj)` must **hard error** rather than triggering implicit full materialization.
       - Default criterion: if the object is file-backed/spilled **or** estimated materialized bytes exceed a configurable ceiling (opt-in setting), exports hard error. Estimated bytes use logical dense bytes (shape product × dtype itemsize).
       - Rationale: deterministic failure is better UX than OS thrash/freezes.
       - Power-user path exists via an explicit kwarg `allow_huge=True` on export entrypoints to intentionally materialize anyway; default is `allow_huge=False`.
     - **Snapshot-backed objects:** exporting to NumPy is a **copy** by default.
       - Rationale: users expect a “real NumPy array”, and snapshot-loaded data is treated as immutable.
   - **Import-from-NumPy policy (to define):**
     - Adopt in place only when the NumPy input is contiguous/aligned and dtype matches PyCauset kind rules; otherwise copy.
     - Default behavior for very large NumPy inputs is to create disk-backed storage rather than reserving equivalent RAM.
     - Optional import kwarg `max_in_ram_bytes` (default `None`): above this cap, force spill/backing file even if contiguity would allow adoption. If `None`, only the “huge → disk-backed” policy applies.
     - Small objects should match NumPy overhead expectations (≥ 0.9× baseline in agreed regimes).

2. **Direct vs streaming decision contract**
   - When kernels are allowed to use OS paging (“direct path”) vs must use explicit streaming.
   - Default routing guidance (user should not need to tune):
     - Stream when operands are file-backed/spilled or estimated materialized bytes exceed a hardware-aware ceiling (implementation may size this from device RAM; threshold is not user-facing by default).
     - Favor streaming for access patterns that are naturally blocked/tilable; allow direct path for small, in-RAM operands with contiguous-friendly layout.
     - Properties/layout may force streaming (e.g., structures implying sparse-ish or strided access that would thrash if paged directly).
   - Ensure decisions are reproducible and testable: emit trace hooks/counters for chosen path, estimated bytes, and why-direct/why-streaming so tests can assert routing.

3. **I/O safety contract**
   - Define what operations are allowed to allocate large intermediates, and which must be blocked or forced into streaming.
   - Block or force-stream any path that would densify/fully materialize spill-backed or huge operands; forbid O(N^2) temporaries unless they stay within a tiling/windowed budget.
   - Mixed-kind or layout-transform intermediates must respect the same budget; if exceeding it, they route through streaming tiling or error explicitly.
   - Surface a deterministic error/warning when an op is rejected for I/O safety; avoid silent slow fallbacks.

4. **File format interoperability contract (import/export/convert)**
   - Establish the supported file-format surface for R1 (prioritized list), without changing the canonical snapshot format:
     - Import: allow PyCauset to read relevant external formats (at least NumPy `.npy` / `.npz`; others as selected).
     - Export: allow writing to external formats for interoperability.
   - Add a **file conversion** API for power users and pipelines:
     - Proposed name: `pc.convert_file(src_path, dst_path, *, dst_format=None, **options)`.
     - Goal: convert `.pycauset` ⇄ (supported formats) without forcing users through manual load + save boilerplate.
   - Optional: define whether a small “pandas interop” surface belongs here (likely optional dependency).

**Definition of done (Phase 0):** A short written contract section in this plan that can be used to reject/accept implementation PRs.

## Phase 1 — Snapshot I/O correctness (all public types) — **Status: Completed**

**Objective:** Ensure `save/load` round-trips are correct across the declared dtype/structure surface.

Deliverables:

- Round-trip tests for:
  - dense matrices/vectors of all public dtypes
  - structured matrices/vectors where applicable
  - view metadata (transpose/conjugation/scalar) and shape
  - properties + cached-derived values (tri-state semantics preserved)
  - spill-backed vs snapshot-backed parity (round-trip behavior identical modulo immutability expectations)
  - mixed device/storage routing does not drop properties or view semantics
- Crash-safety invariants remain true (payload offsets stable, metadata updates do not shift payload).
  - Include a guardrail test that a metadata-only update does not rewrite payload.
  - Include a guardrail that malformed/corrupt metadata fails loudly (not silent partial loads).

**Definition of done (Phase 1):** SRP-style round-trip suite passes for the full public surface, including spill vs snapshot parity and metadata-only updates not touching payload; failures are deterministic and loud.

## Phase 2 — Snapshot I/O performance (large-scale) — **Status: Completed**

**Objective:** Large reads/writes are demonstrably efficient and avoid avoidable extra passes.

Deliverables:

- Benchmarks for:
  - large `save()` throughput (sequential write)
  - large `load()` + first-touch performance
  - repeated loads (OS cache effects acknowledged but measured)
- Verify that “small metadata updates” do not require rewriting payload.

**Definition of done (Phase 2):** Measured throughput is competitive and regressions are detectable.

## Phase 2b — Format interoperability (pragmatic, pipeline-friendly) — **Status: Completed**

**Objective:** Interoperate with the ecosystem without pretending NumPy will become an out-of-core engine.

Deliverables:

- Import from NumPy formats:
  - `.npy` (dense)
  - `.npz` (dense bundles)
- Export to NumPy formats:
  - `.npy` / `.npz` for dense matrices/vectors
- `pc.convert_file(...)` supports `.pycauset` ⇄ `.npy`/`.npz` at minimum.
- If added in R1: pandas interoperability (optional dependency) is explicitly scoped and tested.

Notes (ambition with realism):

- It is reasonable to target an extensive suite of supported formats over time, but we should treat this as a staged effort:
  - **R1 baseline:** a small set of formats with stable, widely-used readers/writers.
  - **R1+ (or optional extras):** additional formats gated behind optional dependencies or a plugin-style layer.
- Additional format targets worth prioritizing (depending on community demand):
  - MatrixMarket `.mtx` (common for sparse matrices)
  - MATLAB `.mat` (common in scientific workflows)
  - A Mathematica-friendly interchange path (exact container TBD; may be easiest via text/CSV or HDF5-based interchange rather than a native `.mx` reader)
  - Parquet / Arrow for pandas-oriented workflows (tabular, columnar)

**Definition of done (Phase 2b):** A minimal set of formats works end-to-end and is documented; conversion failures are deterministic and actionable.

## Phase 3 — Out-of-core kernel I/O strategy — **Status: Completed**

**Objective:** Streaming paths and IO hinting match access patterns and reduce page faults.

Deliverables:

- Identify the “top 3–4” ops to harden for out-of-core behavior (likely `matmul`, `inverse`, `eigval`, and one more depending on current usage).
- Verify that:
  - streaming paths are exercised in tests,
  - IO prefetch/discard hooks are called where intended,
  - direct-vs-streaming selection does not regress.

Working plan (current selection):
- Target ops: `matmul`, `inverse`, `eigval/eig`, top-k eigenvalue (`eigvals_arnoldi`/Arnoldi/Lanczos), and block `cholesky` (fallback to solve if coverage needs broader exposure).
- Streaming/tiling sketch:
  - Matmul: blocked row/col tiles with sequential-friendly read/write; overlap prefetch of A/B tiles with C write-back; GPU path uses pinned host staging for host↔device tiles.
  - Inverse: block-based panel updates (Schur/blocked Gauss-Jordan) with streaming panels; sequential panel read, tiled trailing-update writes; prefer shared tile size heuristic keyed to memory threshold.
  - Eigval/Eig: reduce to banded/tridiag in streaming panels; use chunked workspace for iterative sweeps; avoid full dense materialization when eigenvectors not requested.
  - Cholesky: panel factorization + trailing block updates; stream panels from disk, double-buffer trailing tiles; expose panel tile size knob tied to memory threshold.
- Observability counters/hooks to add: route decision (direct vs streaming), estimated bytes per operand, chosen tile shape, in-flight tile queue depth, prefetch/read/write throughput, page-fault proxy (fallback: OS-reported faults if available), device-idle proxy (GPU: kernel gaps vs H2D/D2H overlap; CPU: worker idle vs IO wait), and a per-op trace tag for why-direct/why-stream.
- Observability scaffolding now records per-op traces (matmul/invert/eigvalsh/eigh) with route decision, estimated bytes, tile shape heuristic, queue depth, page-fault proxy, throughput placeholders, and a trace tag accessible via debug helpers.
- Tests/bench shape: CI-friendly sizes that force streaming via low thresholds; assertions on chosen route and tile shape; smoke benchmarks that report throughput plus counters (no strict perf gates in CI, but numbers logged for regression diffing).

Additional performance direction (explicitly a goal):

- Prioritize **read/write speed under demanding CPU/GPU workloads**.
- Where feasible, overlap compute with I/O (prefetch, write-behind, staging buffers) while keeping behavior deterministic.

How overlap should work (contract-level intent; implementation details may vary):

- **Make access patterns explicit:** for each hardened op, define whether its dominant access is row/column/blocked/strided and choose a tiling that makes disk access as sequential as possible.
- **Pipeline, don’t “read everything then compute”:** treat out-of-core kernels as a steady-state pipeline:
  - stage A: async read (prefetch next tiles)
  - stage B: decode/prepare (layout transforms, dtype conversions if unavoidable)
  - stage C: compute (CPU and/or GPU)
  - stage D: async write (write-behind results, flush checkpoints)
- **Double/triple buffering:** maintain a small ring of tile buffers so compute can proceed while I/O runs.
- **CPU↔GPU staging discipline:** if GPU is involved, use a staging strategy that minimizes stalls:
  - pinned/page-locked buffers where available,
  - async host→device transfers overlapped with kernel execution,
  - device→host write-back overlapped with next read.
- **Backpressure + safety:** if the disk can’t keep up, the system must reduce concurrency/tiling rather than thrash:
  - avoid unbounded queues,
  - avoid pathological random I/O,
  - make fallbacks explicit (e.g., switch to smaller tiles).
- **Deterministic observability:** expose traces/counters that can prove overlap is happening (read throughput, queue depth, page-fault rate, GPU idle time) so regressions are actionable.

Phase 3 success goals (detailed; this is the main bottleneck)

These goals define what “working well” means for large out-of-core runs. They are intentionally measurable, but avoid hard-coding a single numeric target that would be invalid across SSD/HDD/NVMe/network storage.

- **Goal A — No pathological thrashing (stability first):**
  - Large jobs do not enter a “death spiral” of page faults / tiny random reads / runaway memory growth.
  - Queue depths and buffering are bounded (no unbounded in-flight tiles).
  - If the system cannot keep up, it degrades gracefully (smaller tiles / reduced concurrency) rather than stalling unpredictably.

- **Goal B — Sustained throughput is close to hardware limits (performance):**
  - For sequential-friendly kernels (blocked/streamed access), sustained read/write throughput during steady state should be a large fraction of a simple measured baseline for the same backing device (e.g., sequential file read/write micro-benchmark).
  - The kernel should spend most of its time doing useful work rather than waiting on I/O.

- **Goal C — Verified compute↔I/O overlap (not just “async calls”):**
  - Traces show that while stage C (compute) is active, stage A (prefetch) and/or stage D (write-behind) are also active most of the time.
  - On GPU workloads specifically: avoid long GPU idle gaps attributed to input starvation.
  - On CPU workloads: avoid long worker idle gaps attributed to input starvation.

- **Goal D — Access patterns match the plan (predictable I/O shape):**
  - When a kernel is declared “streaming/blocked”, its I/O should be dominated by large, mostly sequential reads/writes (as opposed to many tiny reads).
  - Prefetch/discard hints align with the chosen tiling and do not regress to “hint spam”.

- **Goal E — Deterministic routing and debuggability (engineering reality):**
  - The system can explain *why* it chose direct vs streaming for a given op (inputs, sizes, device, thresholds).
  - A user/dev can answer: “is this op I/O-bound or compute-bound?” from logs/counters.
  - Regressions are actionable: benchmarks include the key counters (throughput, queue depth, page-fault rate proxy, device idle time proxy) alongside wall time.

**Definition of done (Phase 3):** Deterministic tests confirm the right path is taken; large-scale runs show no pathological thrashing.

**Completion notes:**
- IO observability records route decision, estimated bytes, tile heuristic, queue depth placeholder, page-fault proxy, and trace tags for `matmul`, `invert`, `eigh`, `eigvalsh`, and `eigvals_arnoldi`.
- Streaming route now enforces a concrete streaming implementation for matmul (tiled) and streaming fallbacks for invert/eigh/eigvalsh/eigvals_arnoldi, with prefetch + discard hints.
- Threshold controls and trace access/clearing are exposed publicly; CI-friendly tests cover file-backed stand-ins, threshold-driven streaming, and top-k eigen, asserting the streaming implementation tag.

## Phase 4 — Temp storage lifecycle + observability — **Status: Completed**

**Objective:** Temp files and spill behavior are correct, predictable, and diagnosable.

Deliverables:

- Explicitly test:
  - spill-to-`.tmp` behavior under memory pressure
  - cleanup-on-startup and cleanup-on-exit semantics
  - `keep_temp_files` behavior
- Observability:
  - confirm debug traces distinguish compute vs IO events
  - add/confirm lightweight diagnostics for “where is my backing file” and “did this op spill” in logs/traces (not user-facing spam)

**Definition of done (Phase 4):** Temp lifecycle is deterministic and does not leak across runs.

**Completion notes:**
- Temp file tracking now records `.tmp`/`.raw_tmp` creations, exposes tracked files, and cleans recursively across storage roots unless `keep_temp_files` is set.
- Runtime exposes a reusable `cleanup_all_roots(keep_temp_files=...)` used by exit hooks and tests; setting a new backing dir scrubs stale temp files in the target root.
- IO traces include storage summaries (backing/temporary files, roots, spill flag) plus compute vs IO events; diagnostics answer “did this spill?” and “where is the backing file?”
- Tests cover spill-to-temp under low memory thresholds, cleanup-on-set/exit semantics, the `keep_temp_files` toggle, and trace observability of spill + IO events.

## Phase 5 — Documentation and testing deliverables (required) — **Status: Completed**

**Objective:** Make the behavior discoverable and prevent regressions. Follow documentation protocol in [[Documentation Protocol]]

Deliverables:

- Documentation updates:
  - canonical explanation of spill vs snapshot
  - explicit `.tmp`/`.raw_tmp`/`.pycauset` lifecycle
  - clear statement of NumPy conversion semantics and when materialization occurs
- Test coverage:
  - unit tests for round-trips and conversion semantics
  - at least one “large-ish” integration test (size chosen to be CI-friendly)
- Benchmark harness:
  - a small, repeatable I/O benchmark suite with a baseline and regression detection strategy (even if CI initially only reports)

**Definition of done (Phase 5):** Docs + tests + a minimal benchmark harness exist, and failures are actionable.

**Completion notes:**
- Docs: storage guide now documents spill vs snapshot behavior, `.tmp`/`.raw_tmp`/`.pycauset` lifecycle, and explicit NumPy conversion safety (file-backed exports block unless `allow_huge=True`, size ceiling via `set_export_max_bytes`).
- Tests: added conversion policy coverage for file-backed opt-in, snapshot exports, and allow-huge bypass; storage/out-of-core suites remain green.
- Benchmarks: added `benchmarks/benchmark_io_smoke.py` for repeatable save/load smoke throughput with size/MB/s reporting.

## Open questions (design chief)

- Should we provide an explicit opt-in **zero-copy** export for small in-RAM objects (separate API), while keeping `np.asarray` semantics predictable?
- Which external formats are highest priority beyond `.npy`/`.npz` (e.g., MatrixMarket `.mtx`, CSV for debugging, Parquet for pandas workflows)?
- What are the top 2–3 workflows that define “dominate NumPy for humongous arrays” (save/load throughput, matmul, inversion, pipeline conversion, etc.)?
