# R1_NUMPY — Fast NumPy Interop (Release 1 Plan)

**Status:** Plan finalized (contract locked; ready to execute)

**Goal:** NumPy ↔ PyCauset interop is predictable, safe (no surprise huge materialization), and fast enough that mixed workflows are viable.

This plan is the contract for what “NumPy compatibility” means in Release 1 and how it is measured.

Execution note (implementation):

- This plan is **finalized** and ready to begin Phase 1.
- No code should be written until Phase 1 inventory confirms the current surface (this prevents “optimizing the wrong thing”).

---

## 0) Why this exists

PyCauset is positioned as **“NumPy for causal sets”**.

To make that credible:

- Conversion must not be a bottleneck.
- NumPy-first usage should either work (by routing) or fail loudly before a surprise memory blow-up.
- Performance claims must be gated by benchmarks (SRP / Gate E).

Current implementation note (as of 2025-12-27):

- `pycauset.matmul` and `pycauset.invert` already exist as native/endpoint operations with routing (including streaming routes in some cases).
- `pycauset.solve` exists as an endpoint-first baseline (may fall back to `invert(a) @ b`).
- `pycauset.eigh` / `pycauset.eigvalsh` exist but are currently documented as NumPy fallbacks.

---

## 1) Dependencies (read before implementing)

- Roadmap node: [PyCauset Roadmap](TODO.md) → **R1_NUMPY**
- Naming/shape alignment: [NumPy Alignment Protocol](../../project/protocols/NumPy%20Alignment%20Protocol.md)
- Out-of-core conversion policies: [R1_IO Plan](R1_IO_PLAN.md)
- Storage/memory UX rules: [Storage and Memory](../../guides/Storage%20and%20Memory.md)
- Benchmark philosophy: [Support Readiness Framework](SUPPORT_READINESS_FRAMEWORK.md)

---

## 2) Scope (Release 1)

### 2.1 In scope

- Import from NumPy / array-like:
  - `pycauset.vector(np_array)`
  - `pycauset.matrix(np_array)`
- Export to NumPy:
  - `np.asarray(obj)` / `np.array(obj)` via `__array__`
  - `pycauset.to_numpy(obj, ...)`
- Interop ergonomics:
  - mixed operands (NumPy array on either side of operators) where safe
  - limited NumPy override routing (allowlist)
- Performance enforcement:
  - conversion-heavy regimes: **≥ 0.90× NumPy** (<10GB)
  - out-of-core regimes (>RAM): **> 1.00× NumPy baseline**

### 2.2 Non-goals

- N-D arrays (R1 stays 1D vectors + 2D matrices only)
- Full NumPy surface emulation
- “Zero-copy everywhere” (only where stable and low-maintenance)

---

## 3) Terms and invariants (must always remain true)

**Terms**

- **Import:** `numpy.ndarray` → PyCauset vector/matrix
- **Export:** PyCauset object → `numpy.ndarray`
- **Materialization:** allocating a dense in-RAM buffer for all elements

**Invariants**

- No accidental huge materialization (file/spill-backed objects must not implicitly export)
- 2D-only (no silent N-D introduction via interop)
- Predictable copy rules (`copy=True` default)
- Dtype mapping stays consistent with the DType System / promotion rules

---

## 4) Locked decisions (design chief)

1) **No `pycauset.asarray` public API (purge).**

- Import uses `pycauset.matrix(...)` / `pycauset.vector(...)`.
- Export uses `np.asarray(obj)` / `np.array(obj)` and `pycauset.to_numpy(...)`.

2) **Interop should be as broad as possible without diminishing-returns implementation.**

- If a NumPy-first call would force huge materialization, we must either route to PyCauset or fail loudly with an actionable message.

3) **Export copy semantics:** default is `copy=True`; `copy=False` exists when safe.

---

## 5) Contract to implement (Release 1)

### 5.1 Public entrypoints (must be documented)

- Import: `pycauset.matrix`, `pycauset.vector`
- Export: `pycauset.to_numpy`, `np.asarray(obj)` / `np.array(obj)`
- Safety knobs: `pycauset.set_export_max_bytes`
- Disk conversion surface: `pycauset.convert_file`

### 5.2 Rank and dtype boundaries

- Supported ranks: 1D vectors and 2D matrices.
- Unsupported ranks (0D, >2D) must raise a clear error.
- Unsupported dtypes must raise a clear error or follow documented cast rules.

### 5.3 Export safety boundary (no surprise materialization)

- File/spill-backed objects must **hard error** on `np.asarray(obj)` unless explicitly opted in.
- `pycauset.to_numpy(..., allow_huge=True)` is the explicit opt-in.
- `pycauset.set_export_max_bytes(...)` applies to both `np.asarray` and `to_numpy`.

### 5.4 `copy=False` semantics

- `copy=False` returns a **read-only** NumPy view when it can be done without allocation.
- If a view cannot be created safely, `copy=False` must **emit a `UserWarning`** and **fall back to `copy=True`**.

### 5.5 NumPy override protocols

Implement NumPy override protocols in R1 as **allowlist-only routing**, returning `NotImplemented` outside the allowlist.

Initial allowlist (Release 1):

- basic arithmetic ufuncs
- `np.matmul` / `np.dot`
- reductions: `sum`, `mean`

---

## 6) Benchmark gates (Release 1 acceptance criteria)

### 6.1 Conversion gate — “≥ 0.90× NumPy” (<10GB)

**Metric:** throughput ratio for the same semantic boundary.

- $\text{throughput} = \frac{\text{bytes}}{\text{seconds}}$
- Pass: $\frac{\text{throughput}_{pc}}{\text{throughput}_{np}} \ge 0.90$

**Measurement rules**

- Median of 7 runs after 2 warmups
- Payload bytes ≥ 32 MiB (agreed)
- Record CPU/RAM/OS/Python/NumPy versions and thread env vars

**Threading policy**

- For conversion benchmarks, pin thread pools to 1 (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).
- Also run an informational “native-threading” variant (not gated unless promoted).

**Baselines**

- Import (same dtype): baseline is `np.array(arr, copy=True)`.
- Import (cast): baseline is `arr.astype(target_dtype, copy=True)`.
- Export (copy=True): baseline is `arr.copy()` for an equivalent NumPy dense array.
- Export (copy=False): not part of this gate in R1; report separately.

**Required regimes (<10GB)**

- float64 matrix: 2048×2048 (~32 MiB)
- float32 matrix: 4096×4096 (~64 MiB)
- complex_float64 matrix: 2048×2048 (~64 MiB)
- float64 vector: 4,194,304 elements (~32 MiB)
- float32 vector: 8,388,608 elements (~32 MiB)
- bit vector: 268,435,456 bits (~32 MiB packed)
  - baseline uses `np.packbits` / `np.unpackbits` (agreed)

**Pass/fail noise guard**

- Median ratio ≥ 0.90
- Minimum (post warm-up) ratio ≥ 0.85

On failure: file a bug in `tests/BUG_LOG.md` with benchmark output; fix or downgrade with design-chief approval.

### 6.2 Out-of-core gate — “> 1.00× NumPy” (matrices larger than RAM)

This gate targets workloads that **do not fit in RAM**.

**Size rule**

- Payload must be strictly larger than available RAM (e.g., ≥ 1.25×), subject to disk space.
- If RAM detection is unavailable, use a conservative fallback (≥ 12 GiB) and document it.

**Baseline definition (NumPy out-of-core)**

NumPy does not provide canonical out-of-core linear algebra.
The baseline is:

- `numpy.memmap` storage, plus
- a benchmark-harness blocked/tiled algorithm that keeps working sets in RAM and uses NumPy’s in-RAM kernels on tiles/panels.

**Required out-of-core workloads (Release 1)**

1) **Matmul:** float64 blocked matmul
2) **Inverse (full materialization):** float64 out-of-core inverse that produces an explicit $A^{-1}$.

Notes for the inverse workload:

- The output $A^{-1}$ is expected to be written to disk-backed storage (e.g., memmap or PyCauset spill-backed storage). The benchmark must not assume the full inverse fits in RAM.
- Baseline comparison must use a NumPy-first out-of-core approach implemented in the benchmark harness (memmap + blocked algorithm writing $A^{-1}$ to a memmap output).

**Pass/fail rule**

- For each required workload: throughput ratio must be **> 1.00×** vs the defined NumPy memmap+blocked baseline.
- Threading: allow native threading, but record env vars and core count.

### 6.3 Operation benchmarks (tracked alongside interop)

These are tracked because they are core to PyCauset value:

- Matmul (in-RAM): float64 1024×1024 and 2048×2048
- Inverse/solve (in-RAM): float64 1024×1024
- Eigen (in-RAM symmetric): float64 1024×1024 vs `numpy.linalg.eigh`

These do not block R1_NUMPY unless explicitly promoted into the release gate.

---

## 7) Phased execution plan

### Phase 0 — Contract freeze (no semantic changes)

- Verify Sections 5–6 match the intended contract (do not change semantics in Phase 0).
- Verify the out-of-core inverse workload definition: full inverse materialization ($A^{-1}$), written to disk-backed storage.

**Done when:** this plan can accept/reject implementation PRs without re-litigating semantics.

### Phase 1 — Inventory (current behavior map)

- Enumerate which Python-visible types implement `__array__` and what paths they take.
- Enumerate dtype mappings (PyCauset token → NumPy dtype) and document cast rules.
- Identify any gaps between guides and runtime behavior.

**Done when:** a surface map exists and all gaps are listed.

### Phase 2 — Correctness tests (guardrails)

- Add tests for ranks (1D/2D allowed; 0D/>2D errors).
- Add dtype matrix tests (including complex safety and bit packing behavior).
- Add export-guard tests (RAM vs snapshot vs spill/file-backed).
- Add rectangular-matrix conversion tests.

**Done when:** the Phase 2 correctness suite is complete and stable.

### Phase 3 — Import performance (NumPy → PyCauset)

- Ensure a native bulk import path exists for contiguous dense numeric inputs.
- Define behavior for non-contiguous inputs (copy vs error vs materialize policy) and test it.
- Ensure benchmarks for the conversion gate pass for import regimes.

**Done when:** import meets the conversion gate for required regimes.

### Phase 4 — Export performance & safety (PyCauset → NumPy)

- Align `np.asarray(obj)` and `pycauset.to_numpy(obj)` safety rules.
- Implement/optimize export fast paths for RAM-backed objects.
- Implement `copy=False` behavior per contract and test it.

**Done when:** export meets the conversion gate and guardrails.

### Phase 5 — Interop ergonomics (“feels like NumPy”)

- Mixed-operand operator behavior (left/right dispatch) for core operators.
- Implement NumPy override allowlist routing per Section 5.5 and add an interop UX test suite.

**Done when:** selected UX targets behave deterministically and are documented.

### Phase 6 — Benchmark harness & CI integration

- Implement the conversion benchmarks (gated) and out-of-core benchmarks (gated).
- Ensure benchmark output reports ratio vs baseline and key environment metadata.
- Document how to run locally and how failures are logged.

**Done when:** benchmarks run reliably and failures produce actionable artifacts.

### Phase 7 — Documentation footprint

- Ensure API reference pages match runtime behavior for the entrypoints in Section 5.1.
- Update guides:
  - `documentation/guides/Numpy Integration.md`
  - `documentation/guides/Storage and Memory.md`
- Add cross-links (“See also”) from touched docs.

**Done when:** Documentation Protocol checklist is satisfied.

---

## 8) Release gate (what “done” means)

R1_NUMPY is complete when:

- The contract in Sections 5–6 is implemented and tested.
- Conversion gate passes (≥ 0.90× NumPy) for all required <10GB regimes.
- Out-of-core gate passes (> 1.00× NumPy baseline) for matmul **and** inverse/solve workloads.
- Out-of-core safety invariants are enforced (no surprise huge exports).
- Docs match runtime behavior (no phantom docs).
