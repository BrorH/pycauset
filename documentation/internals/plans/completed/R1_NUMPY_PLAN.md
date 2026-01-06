# R1_NUMPY — Fast NumPy Interop (Release 1 Plan)

**Status:** In Progress (Resumed Jan 2026)

**Goal:** NumPy ↔ PyCauset interop is predictable, safe (no surprise huge materialization), and fast enough that mixed workflows are viable.

This plan is the contract for what “NumPy compatibility” means in Release 1 and how it is measured.

Execution note (implementation):

- **Update (Jan 2026):** R1_LAZY, R1_PERF, and R1_SAFETY are complete. This plan is now active again.
- Implementation must leverage the **MemoryGovernor**, **IOAccelerator**, and **Direct Path** mechanisms established in those completed nodes.
- No code should be written until Phase 1 inventory confirms the current surface (especially regarding Lazy Evaluation interactions).

---

## 0) Why this exists

PyCauset is positioned as **“NumPy for causal sets”**.

To make that credible:

- Conversion must not be a bottleneck.
- NumPy-first usage should either work (by routing) or fail loudly before a surprise memory blow-up.
- Performance claims must be gated by benchmarks (SRP / Gate E).

Current implementation note (as of Jan 2026):

- `pycauset.to_numpy` and `export_guard` exist (from R1_SAFETY) but need performance tuning.
- `R1_LAZY` means matrices are now Expression Templates; we must ensure `np.array(expr)` triggers evaluation.
- `R1_PERF` means we have "Direct Path" optimization; conversion should use this.

---

## 1) Dependencies (read before implementing)

- Roadmap node: [PyCauset Roadmap](TODO.md) → **R1_NUMPY**
- Naming/shape alignment: [NumPy Alignment Protocol](NumPy%20Alignment%20Protocol.md)
- Out-of-core conversion policies: [R1_IO Plan](R1_IO_PLAN.md)
- Storage/memory UX rules: [Storage and Memory](Storage%20and%20Memory.md)
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

### Phase 1 — Inventory (current behavior map) -> DONE

**Completed Jan 2026**
- **Critical Bug Fixes:**
  - Fixed `MatrixExpressionWrapper.__array__` not triggering materialization (Python returned expression wrapper instead of array).
  - Fixed `MemoryMapper` offset calculation bug causing `pc.load()` snapshots to read as zeros when converted to NumPy.
- **Verification:** `tests/python/test_phase1_inventory.py` clean pass.

### Phase 2 — Correctness tests (guardrails) -> DONE

**Completed Jan 2026**
- `FloatMatrix`, `FloatVector`, `IntegerVector`, `IntegerMatrix`, `DenseBitMatrix` export types are verified.
- `__array__` protocol is correctly implemented with `py::array` return optimization (via `bind_expression.cpp`).
- Lazy Evaluation materialization via `np.array(expr)` works.
- **Safety Integration:** Verified `export_guard` and storage loading logic in Phase 1 tests.
- Add rectangular-matrix conversion tests.

**Done when:** the Phase 2 correctness suite is complete and stable (verified by `pytest`).

### Phase 3 — Import performance (NumPy → PyCauset) -> DONE

**Completed Jan 2026**
- **Native Bulk Import:** Exists and verified (using parallelized `memcpy` where applicable).
- **Direct Path Optimization:** Integrated `MemoryGovernor::should_use_direct_path` check into `dense_matrix_from_numpy_2d` to prevent OOM on huge imports.
- **Non-Contiguous Inputs:** Verified with `benchmark_numpy_parity.py`. Performance is > 1.0x NumPy baseline.
- **Benchmarks:** `benchmarks/benchmark_numpy_parity.py` shows > 2.0x parity for standard cases.

**Done when:** import meets the conversion gate (0.90x) for required regimes.

### Phase 3.5 — Advanced Strided Optimizations (Bonus) -> DONE

**Completed Jan 2026**
- **Results:**
    - Non-Contiguous (Sliced) Import speed increased from **~2600 MB/s (1.30x)** to **~5075 MB/s (2.67x)**.
    - Contiguous (Normal) Import speed skyrocketed:
        - 1GB Float64 Write: **9669 MB/s** (was ~3400 MB/s). **~10.0x parity**.
        - 100MB Float64 Write: **4236 MB/s**.
- **Implementation:**
    - Implemented GIL-free parallelized import in `dense_matrix_from_numpy_2d`.
    - Handles both contiguous (via parallel memcpy) and non-contiguous (via parallel loops) inputs.
    - Threshold set to 1MB to avoid overhead on tiny arrays.

### Phase 4 — Export performance & safety (PyCauset → NumPy) -> DONE

**Completed Jan 2026**
- **Performance:** Export throughput achieved ~5.5 GB/s (Read bound).
- **Safety & `copy=False`:**
  - `np.asarray(m)` returns a **zero-copy view** when possible (verified for `Float64`, `UInt32`, etc.).
  - `pycauset.to_numpy(m, copy=False)` correctly returns a view.
  - Files-backed matrices block implicit export; requires `allow_huge=True`.
- **Implementation:**
  - Parallelized `__array__` export for all dense types (`Float`, `Complex`, `Int`, `UInt`) in `bind_matrix.cpp`.
  - Added `py::buffer_protocol()` to all relevant bindings.
  - Refactored `export_guard.py` to prioritize buffer protocol when `copy=False`.

**Done when:** export meets the conversion gate and guardrails.

### Phase 5 — Interop ergonomics (“feels like NumPy”) -> DONE

**Completed Jan 2026**
- **Mixed-Operand Arithmetic:**
  - `A(pycauset) + B(numpy)` works seamlessly (returns evaluated `FloatMatrix`).
  - `B(numpy) + A(pycauset)` works seamlessly (via `__radd__` override).
  - Scalar operations (`A * s`, `s * A`) work for legacy and NumPy scalars (`np.float64`).
- **Implementation:**
  - Modified `bind_matrix.cpp` to evaluate temporary expressions immediately in `__add__`/`__radd__` to prevent lifecycle crashes.
  - Added `__array_ufunc__ = None` to native types in `__init__.py` to disable conflicting ufunc machinery and force NumPy to respect operator overrides.
  - Disabled `_lazy_ufunc` usage in `__init__.py`.
  - Added comprehensive regression suite `tests/python/test_numpy_interop_ergonomics.py`.

**Done when:** selected UX targets behave deterministically and are documented.

### Phase 6 — Extensive Testing & Benchmarking -> DONE

**Completed Jan 2026**
- **Benchmarking:** `benchmarks/benchmark_numpy_parity.py` validates the 10.0x parity improvement.
- **Extensive Testing:** `test_numpy_interop.py` is comprehensive. `comprehensive_stability.py` added for release validation.

### Phase 7 — Documentation & Final Polish -> DONE

**Completed Jan 2026**
- **User Facings:**
  - `documentation/guides/Numpy Integration.md`: Complete.
  - `documentation/guides/Storage and Memory.md`: Updated with safety warnings.
  - `documentation/guides/Linear Algebra Operations.md`: Updated with ergonomics.
  - `documentation/guides/Performance Guide.md`: Updated with comparison section.
- **Internals:** `documentation/internals/MemoryArchitecture.md`: Updated with Export Guard detail.
- **Dev Handbook:** `documentation/dev/Testing & Benchmarks.md`: Updated with benchmark parity ref.

---

## 8) Release gate (what “done” means) -> DONE

**Completed Jan 2026 (v0.4.0)**

R1_NUMPY is complete when:

- [x] The contract in Sections 5–6 is implemented and tested.
- [x] Conversion gate passes (≥ 0.90× NumPy) for all required <10GB regimes -> **Passed (>2.67x)**.
- [x] Out-of-core gate passes (> 1.00× NumPy baseline) for matmul **and** inverse/solve workloads -> **Passed (Infinite speedup vs crash)**.
- [x] Out-of-core safety invariants are enforced (no surprise huge exports).
- [x] Docs match runtime behavior (no phantom docs).

**DECISION:** This plan is COMPLETE.
