# Support Readiness Framework (DTypes × Ops × Devices)

**Status**: Active (replaces OPTIMIZATION_CHECKLIST.md)
**Goal**: A scalable, template-driven checklist to keep dtype + operation coverage correct, fast, and consistent across CPU/GPU and RAM/out-of-core.

This document is intentionally **not** a huge per-dtype × per-op grid. Instead it provides:

- A *canonical inventory* of the dtypes and operations the project currently declares/exposes.
- A small set of **gates** (correctness, storage, device routing, lookahead hints, benchmarks) that must be satisfied.
- Two reusable templates:
  - **“Add a dtype” template** (end-to-end integration)
  - **“Add an op” template** (end-to-end integration)

Use this as the “definition of done” for dtype expansion and operation coverage.

---

## 1) Canonical dtype inventory

### 1.1 Frontend dtype tokens (factory layer)

These are the dtype tokens treated as the **frontend contract** for the public constructors/allocators (e.g. `pycauset.matrix(..., dtype=...)`, `pycauset.vector(..., dtype=...)`, `pycauset.zeros(..., dtype=...)`).

Small implementation note: some tokens may be temporarily “declared but not yet wired end-to-end”; when that happens, the system should fail clearly (or route through a compat layer) rather than silently producing a different dtype.

**Real dtypes**

- `bool` / `bit` (packed)
- `int16`
- `int32` (also reachable via `int`)
- `float16`
- `float32`
- `float64` (also reachable via `float`)

**Complex dtypes (same scalar system; not a separate universe)**

- `complex_float16` (first-class; storage-optimized)
- `complex_float32` (a.k.a. `complex64`)
- `complex_float64` (a.k.a. `complex128`)

**Non-goal (by design):** PyCauset does **not** support complex permutations of non-float dtypes (`complex int*` / `complex bit`).

Rationale: in practical workloads, complex-valued linear algebra (eigen/spectral analysis, stable solvers, phase-sensitive kernels) inevitably requires floating-point compute and typically relies on established float/complex-float backends (BLAS/cuBLAS). Supporting complex ints/bits would add significant surface area (promotion, overflow rules, kernels, tests) without enabling meaningful acceleration or common workflows.

Implementation note (small but important): the framework treats these complex permutations as part of the dtype system (i.e. they “exist” conceptually), but they are not yet wired end-to-end in the codebase.

**How `float` works (must be explicit)**

- `float` means `float64` (storage and compute unless an op’s policy explicitly differs).
- `float32` is the explicit smaller float storage/compute dtype.
- Mixed-float behavior is **underpromotion by default**: e.g. `float32` + `float64` operations select `float32` unless an op’s promotion policy says otherwise (warnings may apply).

### 1.2 Core scalar model + C++ `DataType` snapshot

PyCauset’s scalar system is best understood as a **base dtype** plus a small set of orthogonal flags.

- Base kind: `bit | int | float`
- Base widths: `1` for bit; `16/32/64` (and others as added) for int/float
- Flags: `complex` (and later `unsigned` for integers)

Under this model, `complex_float64` is “`float64` + `{complex}`”, and similarly for other float base dtypes.

**C++ implementation snapshot**

Authoritative source: `include/pycauset/core/Types.hpp`.

- `BIT`
- `INT16`
- `INT32`
- `FLOAT16`
- `FLOAT32`
- `FLOAT64`

**Policy:** If a dtype exists in `DataType`, it must have an explicit status in this framework:
- **Public** (reachable via Python factories), or
- **Internal-only** (reachable only via internal constructors/bindings), or
- **Planned** (declared but intentionally not wired end-to-end yet).

**Complex representation rule:** complex is a flag/permutation over float base dtypes. Storage strategies:

- Complex floats (`complex float32/float64`) should use true complex numeric storage where possible (BLAS/cuBLAS path).
- Complex float16 is first-class for storage/bandwidth wins at very large scale; solvers may upcast internally (e.g., compute in float32) where required for stability.

Reference implementation plan: `documentation/internals/DTYPE_COMPLEX_OVERFLOW_PLAN.md`.

---

## 2) Canonical operation inventory

### 2.1 Frontend (LinearAlgebra) operations

Authoritative sources:
- `include/pycauset/math/LinearAlgebra.hpp`
- `src/math/LinearAlgebra.cpp`

**Matrix × Matrix**
- `add(a, b)`
- `subtract(a, b)`
- `elementwise_multiply(a, b)`
- `matmul(a, b)`

**Vector × Vector**
- `add_vectors(a, b)`
- `subtract_vectors(a, b)`
- `dot_product(a, b)`
- `cross_product(a, b)`

**Matrix/Vector mixed**
- `matvec(a, x)`
- `vecmat(x, a)`
- `outer_product(x, y)`

**Special**
- `compute_k_matrix(...)`

**Definition of “all operations” (minimum):** everything declared in `include/pycauset/math/LinearAlgebra.hpp`.

### 2.2 Device interface operations

Authoritative source: `include/pycauset/compute/ComputeDevice.hpp`.

**Note:** The device layer is the *implementation contract*. If an operation exists here, it must have:
- CPU correctness (required)
- GPU coverage (optional, but must be explicitly routed/blocked)

### 2.3 Object protocol (required for any public dtype)

Independently of “math ops”, a dtype is not considered **publicly supported** unless its matrix/vector types satisfy:

- Construction (size-based + data-based)
- Element access + mutation (`get`, `set`, `__getitem__`, `__setitem__` as applicable)
- Complex safety: complex-valued objects must not silently drop imaginary parts (real-only access must be guarded or error)
- Transpose behavior (view vs materialize policy is explicit)
- Scalar handling (where supported) and dtype reporting
- Persistence hooks (`copy_storage`, `_from_storage`) and round-trip load/save
- NumPy interop where available (e.g., `asarray` fast-path or explicit conversion path)

---

## 3) Storage + persistence invariants

### 3.1 In-memory storage expectations ("optimal storage")

“Optimal storage” is defined by the smallest representation that preserves semantics *and* supports the required access patterns for the operation set.

Minimum expectations by kind:

- **bit**
  - Use packed storage for dense/triangular bit matrices and bit vectors.
  - “Numeric” ops must explicitly widen (and document the widening), or be error-by-design.

- **int16 / int32**
  - Dense storage is row-major contiguous.
  - Mixed-precision integer ops must either:
    - use dedicated integer kernels, or
    - widen via documented promotion rules.

- **float32 / float64**
  - Dense storage is row-major contiguous.
  - Large-size defaults may prefer float32 for storage efficiency when policy says so.

- **complex (flag/permutation)**
  - Supported complex dtypes are complex floats only.
  - Storage is either true complex float storage (performance path) or two-plane float16 storage for `complex_float16` (scale-first / bandwidth path).

### 3.2 On-disk format expectations

Persistence is ZIP + raw storage payload (often memory-mapped directly out of the zip member).

**Invariants**
- `metadata.json` must record at least:
  - `matrix_type`
  - `data_type`
  - dimensions + seed + scalar + transpose flag
- `data.bin` is the raw storage payload used by `_from_storage(...)` to memory-map.

**Complex payloads (current):** `complex_float16` uses two-plane storage in-memory, but persistence round-trips via a **single raw payload** (`data.bin`) containing both planes contiguously. `metadata.json` identifies the dtype as `complex_float16` and normal shape/layout fields.

**Complex payloads (optional future enhancement):** multiple payload members inside the zip (e.g. `data_real.bin` + `data_imag.bin`) could be added later for ease of inspection/tooling, but are not required for correctness.

**Definition of done:** saving + loading must round-trip for each (dtype, structure) that is public.

---

## 4) Cooperative Compute Architecture (CCA) / Lookahead protocol gate

Authoritative source: `documentation/internals/CooperativeArchitecture.md`.

**Rule:** For operations that stream or stride through persistent storage (especially matmul-like kernels), the solver must emit `MemoryHint`s before heavy reads.

Minimum expectations:
- Emit `Sequential` hints for row-major streaming reads
- Emit `Strided` hints for transposed/column-like access
- Tests must cover that hints are emitted (at least for CPU matmul path)

---

## 5) Device coverage + routing rules

- CPU correctness is mandatory.
- GPU coverage is optional **per operation** and **per dtype/structure subset**.
- Device selection is expected to be **automatic by default**: choose CPU vs GPU based on hardware capability and a lightweight micro-benchmark/heuristic.
- If GPU does not support a case, the system must do exactly one of:
  - route to CPU in `AutoSolver` (explicit CPU-only policy for that op/case), or
  - throw a clear error when GPU is selected/forced.

“Silent wrong answers” is the only unacceptable outcome.

### 5.1 Current implementation note (AutoSolver)

The current design already aligns with the “measure and choose” policy:

- `AutoSolver` runs a small matmul micro-benchmark when a GPU device is enabled.
- Dispatch uses thresholds + the measured speedup factor to prefer CPU when GPU would be slower.

Future direction (not implemented yet): cooperative CPU+GPU *tandem execution* for a single operation.

---

## 6) Gates (what must be true before claiming “supported”)

Treat each gate as a checklist you can apply to either “a new dtype” or “a new op”.

### Gate A — Correctness
- Small deterministic cases (hand-checkable)
- Randomized cases (property-style)
- Cross-dtype cases (promotion/overflow/underpromotion)
- Complex-variant coverage for public complex dtypes (and “error-by-design” assertions where complex closure is planned)
- “Error-by-design” cases are explicit and tested

### Gate B — Storage + persistence
- Round-trip save/load for dense + triangular + vector variants that are public
- Out-of-core path is exercised (memory-mapped load + compute)

### Gate C — Routing + device policy
- CPU path exists
- GPU path is implemented or explicitly blocked/routed
- Behavior is consistent across frontend (LinearAlgebra) and Python surface

### Gate D — CCA lookahead hints
- Operation emits hints for persistent operands when it has a predictable access pattern

### Gate E — Benchmarks (vs NumPy)
- In-memory benchmarks vs NumPy for at least one representative shape regime
- Disk-backed benchmark that exercises paging behavior (large mmap-backed payload)

---

## 7) Template: Adding a new dtype (end-to-end)

Fill this template for each new dtype you add.

### 7.1 Declare and normalize
- [ ] Add/confirm `DataType` enum value
- [ ] Define dtype token(s) for Python
- [ ] Implement dtype normalization (accept `pc.*`, `np.*`, and case-insensitive strings)

### 7.2 Storage types and matrix/vector coverage
- [ ] Dense matrix type exists
- [ ] Triangular matrix type exists if required by the structure policy
- [ ] Vector type exists
- [ ] Identity/Diagonal participation is defined (supported or explicitly excluded)

### 7.3 Factory + persistence
- [ ] `ObjectFactory` supports create/load/clone
- [ ] Python `save/load` metadata mapping exists
- [ ] `_from_storage` works for mmap-backed data

### 7.4 Operations
For each canonical operation group (Section 2):
- [ ] Explicit dtype behavior statement (promotion, overflow, underpromotion)
- [ ] CPU correctness
- [ ] GPU policy (supported or blocked)

### 7.5 Tests + benchmarks
- [ ] Gate A–E satisfied

---

## 8) Template: Adding a new operation (end-to-end)

Fill this template for each new operation you add.

### 8.1 Define the contract
- [ ] Operand ranks supported (M×M, V×V, M×V, V×M)
- [ ] Shape rules and error messages
- [ ] Result dtype + structure rules (promotion + matrix-type promotion)

### 8.2 Wire end-to-end
- [ ] `ComputeDevice` interface method added
- [ ] CPU implementation (`CpuSolver`) + passthrough (`CpuDevice`)
- [ ] `AutoSolver` routing policy (CPU-only or GPU-enabled)
- [ ] Frontend wrapper in `LinearAlgebra` (allocation + dispatch)
- [ ] Python bindings + Python API

### 8.3 Coverage axes (Protocols.md)
- [ ] Operand rank
- [ ] Scalar kind + flags (bit/int/float; complex/unsigned if applicable)
- [ ] Structure/storage (dense/triangular/identity/diagonal/unit-vector)
- [ ] Device coverage (CPU required; GPU optional)
- [ ] Python surface
- [ ] Documentation + tests

### 8.4 CCA lookahead
- [ ] Emit memory hints for persistent operands when applicable

### 8.5 Tests + benchmarks
- [ ] Gate A–E satisfied


