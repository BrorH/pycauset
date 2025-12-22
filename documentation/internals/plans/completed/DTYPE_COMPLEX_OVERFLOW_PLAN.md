# DType / Complex / Overflow Plan (Implementation)

**Status (2025-12-16):** This implementation plan is **complete**. Phase 1 complete; Phase 2 complete (int8/int16/int32/int64 + uint8/uint16/uint32/uint64 + float16 end-to-end); Phase 3 complete (**complex floats are now first-class end-to-end on CPU** for the current core-op surface); Phase 4 complete (support matrix declared + enforced in tests/tools). Optional backlog items are still listed below.

## Scope update (2025-12)

This plan originally sketched “complex permutations for all base dtypes” (including `complex int*` and `complex bit`).

**Current project direction:** complex support is limited to **complex floats** only (`complex_float16`, `complex_float32`, `complex_float64`). Complex permutations of non-float dtypes (`complex int*` / `complex bit`) are a **non-goal by design** due to high implementation surface area (promotion/overflow/kernels/persistence/tests) with low practical payoff for PyCauset’s workloads.

As a result:
- Phase 2 includes first-class `float16` as a general dtype, plus the full signed/unsigned integer width set.
- Phase 3 (complex) should be interpreted as “complex float integration”, with `complex_float16` implemented after `float16` readiness.

## Phase completion status

- **Phase 0 — Documentation & policy grounding:** Complete
- **Phase 1 — Centralize promotion + overflow policies:** Complete
- **Phase 2 — Scalar system expansion:** Complete (int8/int16/int32/int64, uint8/uint16/uint32/uint64, float16 end-to-end through factories/promotion/CPU dispatch/persistence/bindings/NumPy for the core op surface)
- **Phase 3 — Complex system integration:** Complete (**complex_float16/32/64 are first-class dtypes** through factories/promotion/CPU dispatch/persistence/bindings/NumPy for core ops)
- **Phase 4 — Coverage enforcement:** Complete (support matrix declared + enforced in tests/tools)

This file is an implementation plan. The authoritative dtype behavior documentation lives in:

- `documentation/internals/DType System.md`

User-facing summary of what shipped in Release 1:

- [[guides/release1/dtypes.md|Release 1: DTypes (what shipped)]]

## 0) Problem statement

PyCauset supports several fundamentally different scalar/storage types (bit-packed `bit`, integers, floats) plus a partially-separate complex system. Adding a new operation currently requires touching multiple layers and remembering many dtype-specific corner cases:

- type/promotion rules are split between global helpers and per-op frontends,
- CPU kernels often dispatch on “result dtype” and omit some types,
- complex numbers are currently not a first-class `MatrixBase` dtype and therefore drift from the main dispatch/type-resolution path,
- missing coverage is easy to ship because there is no single enforceable “support matrix”.

This document proposes a new, centralized dtype architecture that:

- makes complex floats first-class in the scalar type system,
- adds multiple integer widths (signed/unsigned),
- defines explicit promotion + overflow policies,
- keeps the “anti-promotion / smallest type” ethos,
- keeps performance and out-of-core constraints as first-class concerns.

## 1) Key constraints (from project philosophy + recent decisions)

- **Scale-first:** matrices may be 100GB+; memory blowups are unacceptable.
- **Underpromotion default:** when PyCauset underpromotes, it means **compute and result storage both use the smallest selected dtype**.
- **No silent widening for accuracy:** no hidden “compute in float64 then downcast” in the default path.
- **Bit matrices are numeric for arithmetic ops:** treat `bit` values as 0/1 numeric values for arithmetic ops (e.g., `+`, `*`, `dot`, `matmul`). Bitwise ops are explicit and must preserve bit-packed storage.
- **Overflow behavior:** integer overflow is a runtime error. PyCauset does **not** auto-promote to avoid overflow.
- **Overflow warning:** for large integer matmul, run a worst-case bound preflight and emit a warning when overflow looks plausible.
- **Complex floats are first-class:** complex support is limited to float base dtypes.
  - `complex_float32` / `complex_float64` are BLAS-backed where applicable (native complex types `complex64` / `complex128`).
  - `complex_float16` is implemented as a first-class dtype using a two-plane float16 storage model.
- **Complex non-floats are a non-goal:** `complex int*` / `complex bit` are intentionally unsupported to avoid a large promotion/overflow/kernel/persistence surface area with low payoff.
- **Fundamental-kind rule (bit/int/float):** PyCauset never “promotes down” across fundamental kinds. If an operation mixes kinds, the result kind is the higher kind required by the operation’s semantics.

## 2) Terminology

- **Scalar type:** the per-element numeric type (bit/int/float plus width and flags).
- **Matrix structure:** dense/triangular/symmetric/etc. (storage layout and indexing constraints).
- **Operation (op):** add/subtract/elementwise multiply/matmul/inverse/eigvals/etc.
- **Promotion policy:** rules for selecting result dtypes for mixed-input ops.
- **Overflow policy:** what happens when integer arithmetic overflows.

## 3) Proposed scalar type model (flags/permutations)

Represent scalar types as:

- `kind`: `bit | int | float`
- `width_bits`: for int/float (8/16/32/64), and 1 for bit
- `flags`: a small set of orthogonal modifiers
  - `complex` (supported for float scalar types only)
  - `unsigned` (valid only for `int`)

Examples:

- `bit` = (bit, 1, {})
- `int16` = (int, 16, {})
- `uint16` = (int, 16, {unsigned})
- `float16` = (float, 16, {})
- `complex float16` = (float, 16, {complex})
- `float32` = (float, 32, {})
- `complex float32` (`complex64`) = (float, 32, {complex})
- `float64` = (float, 64, {})
- `complex float64` (`complex128`) = (float, 64, {complex})

### Supported scalar set (initial target)

- bit
- int8/int16/int32/int64
- uint8/uint16/uint32/uint64
- float16/float32/float64
- complex_float16/complex_float32/complex_float64

## 4) Complex implementation strategy

### 4.1 Complex floats (performance path)

- Implement `complex_float32` (`complex64`) and `complex_float64` (`complex128`) as true complex numeric types.
- Prefer BLAS-backed complex GEMM where applicable.

### 4.2 Complex float16 (two-plane storage path)

- Represent `complex_float16` as two float16 planes (real + imag).
- Motivation: there is no ubiquitous, efficient “native complex half” representation across the stack, and forcing complex-half into complex-float32 would violate the “smallest type” ethos.
- Persistence must round-trip as a single complex dtype (one logical object, two payload planes).

### 4.3 Explicit non-goals

- Complex permutations of non-float dtypes (`complex int*`, `complex bit`) are intentionally out of scope.
- If/when we ever revisit this, it must be driven by concrete workloads and come with a scoped support matrix (ops × dtype) rather than a blanket “closure” rule.

This plan does not assume automatic widening in integer matmul. Under the current policy:

- integer overflow throws, and
- the system does not silently widen storage to avoid overflow.

If we ever decide that a particular op’s *semantic* result dtype must be wider (e.g., a count-producing op), that must be a named, explicit promotion rule and must be documented as semantics, not an overflow workaround.

## 5) Promotion policy (centralized, op-specific)

Create a single authoritative table/function:

- `resolve_result_scalar(op, a_scalar, b_scalar) -> scalar`
- `resolve_result_structure(op, a_structure, b_structure) -> structure`

Design principles:

- Default to the smallest dtype that can represent the result per op semantics.
- Mixed float precision underpromotes by default (compute+store in the smaller float), with a configurable option to promote instead.
- Complex is a flag: complex-ness is preserved unless an op is explicitly defined to drop it.
- Unsigned is preserved where meaningful; if an op can generate negatives, rules must define whether to promote to signed or throw.

### 5.1 Fundamental kinds (bit / int / float) and “no promote down”

PyCauset distinguishes three fundamental kinds:

- `bit` (bit-packed boolean storage; special rules allowed)
- `int` (signed/unsigned integers)
- `float` (float16/float32/float64)

Rules:

1) **No promote down across kinds.** If kinds differ, the result kind cannot be the “lower” kind.
2) **When a float participates, the result kind is float.** Example: `matmul(bit, float64) -> float64`.
3) **When only integers/bits participate, the result kind is integer unless the op is explicitly bitwise.**
4) **Underpromotion applies within a kind, not across kinds.** Example: `matmul(float32, float64) -> float32` by default.

This strikes a balance:

- it preserves the “smallest type” ethos where it is meaningful (within float precision),
- it avoids absurd outcomes like underpromoting a float computation to bit storage,
- it keeps `bit` special (bitwise ops remain bitwise; numeric ops may change kind).

### 5.2 Bit is special (scale-first exceptions)

Bit matrices/vectors are used to represent large binary structures (e.g., spacetime relations) where the storage is often 10s–100s of GB.

As a result:

- **Bitwise ops** (e.g., NOT/AND/OR/XOR) should preserve `bit` and stay bit-packed.
- **Numeric ops** that inherently create non-binary results (e.g., `bit + bit`, `matmul(bit, bit)` producing integer counts) may require widening to `int` or `float`.

For such numeric ops, widening can be prohibitively expensive. Therefore, for `bit` we allow explicit, op-specific behavior:

- supported with a documented widening result kind, **or**
- **error-by-design** unless the user explicitly requests a widened dtype.

The support matrix must record which choice is made for each op.

Config hooks:

- `promotion_policy.float_mixed`: `underpromote_warn` (default) | `promote` | `underpromote_no_warn`

Warning controls (exact API TBD, but must exist):

- `warning_policy.float_underpromotion`: on by default when `promotion_policy.float_mixed=underpromote_warn`
- `warning_policy.int_reduction_acc_widen`: on by default; emitted when `dot`/`matmul` widens the accumulator dtype
- `warning_policy.int_overflow_risk_preflight`: on by default for “large” integer matmul; emitted when conservative bounds indicate plausible overflow in the requested output dtype

## 6) Overflow policy

### 6.1 Runtime behavior

- Overflow is a hard error.
- PyCauset does not auto-promote storage to avoid overflow.

### 6.1.1 Why this focuses on integer overflow (and not float overflow)

Floating-point overflow is real (e.g., float32 can overflow to `+inf`), but it behaves differently:

- IEEE-754 overflow typically becomes `inf` (and may raise a floating-point flag), which then propagates.
- This is often detectable after-the-fact (e.g., `isfinite` checks), whereas integer overflow in C++ can be undefined behavior or silent wrap depending on the implementation.

Policy-wise:

- For integers: overflow must throw (no silent wrap).
- For floats: overflow results in `inf`/`nan` according to IEEE-754; optional “finite-check” validation can exist as a debug/strict mode, but it is not the default because scanning 100GB+ outputs is expensive.

### 6.2 Preflight warning for large integer matmul

For integer matmul (and potentially some other high-risk ops), run a cheap preflight to estimate overflow risk:

1) sample blocks/rows to estimate `max_abs(A)` and `max_abs(B)` (including scalar metadata factors if they apply)
2) compute a conservative bound:

$$\max |C_{ij}| \le K \cdot \max|A| \cdot \max|B|$$

Where $K$ is the inner dimension (for square matmul, $K=N$).

If the bound approaches/exceeds the target dtype max value, emit a warning:

- `PyCausetWarning: matmul(<lhs_dtype>, <rhs_dtype>) may overflow <out_dtype> (conservative bound). Consider requesting a wider output dtype or scaling.`

Notes:

- This is a heuristic. It should warn on risk; it does not guarantee overflow will happen.
- It avoids inner-loop overflow checks in the performance-critical kernel.

Documentation requirement:

- Add an “Overflow” section/doc describing the policy, the preflight warning, and user mitigations.

### 6.3 Reduction-aware accumulator width (dot/matmul) + required warning

Some integer reductions (especially `dot`/`matmul`) can overflow the *accumulator* even when inputs are representable and the *requested output dtype* is unchanged.

To keep integer math defined and to uphold “overflow throws” without requiring expensive per-multiply-add overflow checks inside the hot loop, PyCauset uses a **reduction-aware accumulator width** for integer reductions.

Key clarifications (scale-first):

- This rule is about the **accumulator dtype** (compute registers / local scratch), not about materializing inputs.
- In particular, `bit` inputs stay **bit-packed**; `matmul(bit, int16)` does not expand the `bit` matrix to `int32` elements.
- This rule does **not** silently widen the *result storage dtype*. If the user requests `int16` output, the result is stored as `int16` and overflow remains a hard error (typically detected at the final cast from the wider accumulator).

#### 6.3.1 Accumulator-width selection (deterministic / conservative)

For `matmul`/`dot` over integer kinds (including `bit` treated as numeric 0/1), choose an accumulator dtype wide enough that the worst-case bound for the reduction fits.

For `C = A @ B` with inner dimension `K`:

- Use a conservative magnitude bound based on dtype limits (no sampling required):

$$\max |C_{ij}| \le K \cdot \max|A| \cdot \max|B|$$

- For `bit`, $\max|A| = 1$.

For integer dtypes, $\max|A|$ and $\max|B|$ may be taken as the **maximum representable magnitude** for their dtypes (e.g., for `int16`, 32767). This is conservative and ensures accumulator selection is correctness-preserving without needing an extra pass over out-of-core data.

This is intentionally conservative: it is designed to be computed cheaply and to be correct without relying on probabilistic assumptions.

Optionally (future optimization): when it is cheap relative to the matmul itself and does not force an extra out-of-core pass, tighten the bound using **exact streaming summaries** such as row popcounts for `bit` and per-column max-abs for the integer operand.

#### 6.3.2 User-visible warning (required)

Whenever the chosen **accumulator dtype** is wider than what a reader would naively expect from the inputs (e.g., `matmul(bit, int16)` accumulating into `int32`), PyCauset must emit a warning so users understand what is happening.

The warning must include:

- operation name (e.g., `matmul` / `dot`)
- lhs dtype and rhs dtype
- chosen accumulator dtype
- output storage dtype (explicitly stating whether it changed or not)
- reason (reduction-aware widening to keep integer overflow defined)

Suggested warning text (exact wording not required, but content is):

- `PyCausetWarning: matmul(bit, int16) will accumulate in int32 (reduction-aware integer width). Output dtype remains int16; overflow still throws on cast. Bit input remains bit-packed (no materialization).`

Noise control:

- Warn once per call site (or once per unique `(op, lhs_dtype, rhs_dtype, out_dtype, acc_dtype)` tuple) to avoid spam.
- Provide a user-facing way to silence/route warnings (Python `warnings.warn(...)` category, and/or a context flag).

## 7) Enforceable op coverage (“support matrix”)

Introduce an explicit coverage matrix that enumerates for each operation:

- required scalar families (bit/int/float + complex)
- supported widths
- supported structures (dense/triangular/symmetric/etc.)
- required behaviors (defined, error-by-design, or unimplemented)

Goal:

- When a new op is added, missing dtype coverage becomes a failing test/tool run, not a surprise at runtime.

## 8) Implementation sequence (phased)

### Phase 0 — Documentation & policy grounding (Complete)
- Update project philosophy to explicitly define underpromotion and overflow behavior.
- Add roadmap entry for multi-int widths + unsigned.
- Add this plan doc.

### Phase 1 — Centralize promotion + overflow policies (Complete)
- Single promotion resolver per op.
- Central overflow policy + preflight warning for integer matmul.
- Reduction-aware accumulator width for integer `dot`/`matmul` + required user warning when accumulator widens.
- Add mandatory tests for resolver correctness, warning emission, and reduction accumulator selection (see “Mandatory tests”).

### Phase 2 — Scalar system expansion (Complete)
- Add integer widths + unsigned.
- Ensure constructors, IO, numpy interop, and basic ops exist.

### Phase 3 — Complex system integration (Complete)
- Core complex-float dtype integration is implemented (CPU + persistence + Python/NumPy for key ops).
- See “Phase 3 — Complex system integration (Detailed)” in Section 8.1.

### Phase 4 — Coverage enforcement (Complete)
- Support matrix exists and is executed by unit tests and a dev checker tool, so declared support can’t silently regress.

## 8.1) Phase 3 — Complex system integration (Detailed)

**Objective:** Make complex **float** dtypes first-class and integrate them into the same end-to-end pipeline as real dtypes (frontend allocation → promotion resolver → CPU/GPU dispatch → persistence → Python).

**User-facing requirement:** complex float dtypes must behave like normal dtypes on the frontend. For example, `pc.complex_float16` (or equivalent public token) must be a valid `dtype=` argument to `Matrix`/`Vector` factories.

**Scope for Phase 3:** expand complex support to **float base dtypes only**:

- `float16` → `complex_float16` (two float16 planes)
- `float32` → `complex_float32` (a.k.a. `complex64`)
- `float64` → `complex_float64` (a.k.a. `complex128`)

**Out of scope:** complex permutations of non-float dtypes (`complex int*`, `complex bit`).

### 3.x Phase 3 status update (2025-12-16)

Completed in the current codebase:

- First-class complex float dtypes exist end-to-end: `complex_float16/32/64`.
- Storage:
  - `complex_float32/64`: dense storage uses native complex element types.
  - `complex_float16`: two-plane float16 storage (real+imag) for both matrices and vectors.
- Dispatch/promotion:
  - promotion resolver supports complex results for matmul/add/sub/elementwise, plus dot/matvec/vecmat/outer.
  - CPU solver contains complex implementations for dot/matvec/vecmat/outer and vector elementwise/scalar ops.
- Python/NumPy/persistence:
  - dtype tokens + factory inference + `np.array(...)` interop + container persistence round-trip.
  - dot returns Python `complex` when either operand is complex.

Optional backlog (not required for plan completion):

- Ensure solver/eigensystem outputs use first-class complex dtypes end-to-end (no parallel complex object model).
- BLAS/cBLAS complex GEMM path for dense complex matmul on CPU (and GPU complex where applicable).
- Expand complex coverage across additional operations beyond the current core set.

### 3.0 Replace legacy `ComplexMatrix` / `ComplexVector` (compat layer)

Current state (updated 2025-12-16):

- First-class complex float matrices/vectors now exist as `MatrixBase`/`VectorBase` dtypes (`complex_float16/32/64`).
- The legacy `ComplexMatrix` / `ComplexVector` concept may still exist in some solver/eigensystem return paths.
  That legacy path is now considered technical debt (it drifts from the first-class dtype pipeline).

Plan (still valid):

- Ensure any remaining solver/eigensystem paths route through first-class complex dtype matrices/vectors.
- Long-term goal: complex is a normal `MatrixBase`/`VectorBase` dtype, so `LinearAlgebra` and `ComputeDevice` don’t need a parallel complex universe.

Frontend contract note:

- Provide explicit dtype tokens for complex floats (at minimum: `complex_float16`, `complex_float32`, `complex_float64`).
- These tokens must normalize through the same dtype normalization funnel as real dtypes and participate in the same factory code paths.

### 3.1 Make “complex” first-class in the scalar type model

Requirement: represent scalar types as `(kind, width_bits, flags)` where `flags` includes at least `{complex, unsigned}`.

Implementation direction:

- Introduce a `ScalarType` descriptor (or equivalent) that can represent:
  - base dtype (`float16/float32/float64`)
  - flags (`complex`)
- Plumb this through the type-resolution path so promotion is defined as:
  - `resolve_result_scalar(op, a_scalar, b_scalar) -> scalar`

Design constraint (to match the frontend requirement):

- Even though complex can be *represented* as `(base_dtype + complex flag)`, it must be treated as a **distinct dtype identity** for:
  - promotion resolution,
  - dispatch selection,
  - persistence metadata,
  - and the support-matrix enforcement (coverage must be tracked per complex permutation).

Back-compat note:

- The existing `DataType` enum can remain as a *legacy base-type id* during migration, but Phase 3 must ensure complex-ness is not “out-of-band” anymore.

### 3.2 Storage strategy for complex (by base kind)

We intentionally use **two different representations** depending on the float width, to balance performance and scale-first storage efficiency.

#### 3.2.1 Complex floats (performance path)

- `complex_float32` (`complex64`) and `complex_float64` (`complex128`) are true complex numeric types.
- Implement dense complex storage as contiguous `std::complex<float>` / `std::complex<double>` (or ABI-compatible equivalent).
- Route matmul to BLAS complex GEMM where possible.
- GPU: use cuBLAS complex GEMM when available.

#### 3.2.2 Complex float16 (two-plane storage path)

- Represent `complex_float16` as **two float16 planes** of equal shape:
  - real plane: `float16`
  - imag plane: `float16`
- Motivation: avoid forcing half-precision complex values into float32 complex storage, and avoid depending on a non-portable “native complex half” ABI.

Important clarification:

- “Two-plane storage” is an implementation detail. The object is still a single complex-typed matrix/vector from the API perspective, and it must round-trip via persistence as a complex dtype (not as two unrelated real objects).

### 3.3 First-class complex matrices/vectors in the core object model

**Hard requirement:** complex objects must participate in factories, persistence, and dispatch the same way other dtypes do.

Minimum deliverables:

- A `MatrixBase`-derived complex matrix implementation for:
  - `complex_float32` / `complex_float64` (dense)
  - `complex_float16` (two-plane storage)
- A `VectorBase`-derived complex vector implementation (same split).

Interface hazards to address explicitly (to avoid “biting us later”):

- Many existing code paths use `get_element_as_double(...)`. For complex dtypes, this must never silently drop the imaginary part.
  - Either implement `get_element_as_double` as a hard error for complex matrices, or ensure it is only used behind a “real-only” guard.
  - Complex-aware paths must use `get_element_as_complex(...)`.
- `ComputeDevice::multiply_scalar` currently takes `double`; Phase 3 must define the complex-scalar story:
  - either add complex-scalar device entry points, or
  - restrict complex-scalar multiply to frontend methods that dispatch to complex kernels.

### 3.4 Operation coverage policy for complex

Phase 3 does **not** require “every op supports every complex dtype” on day one, but it must make coverage enforceable:

- For each op in the canonical LinearAlgebra surface (at least `LinearAlgebra.hpp`):
  - declare complex propagation rules (preserve complex, drop complex, or error-by-design)
  - declare result dtype selection rules (including for `bit` special cases)
- Ensure the resolver has explicit rows for complex permutations.

Coverage principle (mathematical independence):

- Complex permutations must be treated as separate coverage targets even when they reuse plane-wise kernels.
- “Works because it decomposes into two real ops” is not a substitute for tests: each complex dtype/op combination must be explicitly tested (or explicitly error-by-design with a stable error).

Specific expectations:

- `complex_float32/complex_float64`:
  - `add/sub/elementwise/matmul` must work on CPU.
  - GPU support is optional, but routing must be correct (fallback to CPU when unsupported).
- `complex_float16`:
  - `add/sub/elementwise/matmul` must work on CPU.
  - if implemented via two-plane arithmetic, correctness must be validated vs NumPy complex computations.

### 3.5 Persistence format for complex

Current implementation note (updated 2025-12-16):

- `complex_float16` uses a two-plane in-memory layout (real + imag), but is persisted as a **single contiguous raw payload** containing both planes back-to-back.
- Typed metadata records the dtype identity (`complex_float16`) and the normal shape/layout fields; there is no need for multi-member payloads to round-trip correctly.

Future option (not required for correctness):

- Multi-member payloads could still be introduced later for tooling/inspection convenience, but would be an on-disk format enhancement rather than a correctness requirement.

### 3.6 GPU/CPU selection policy

Match project intent:

- Default behavior: benchmark/poll hardware once, then pick the fastest device.
- If GPU does not support a dtype/op/structure, fall back to CPU.
- Avoid exploding “one kernel per infinitesimal device” by using:
  - a small set of coarse regimes (dtype/shape thresholds)
  - a micro-benchmark-derived speedup factor

### 3.7 Tests (keep the explosion under control)

The only way this stays maintainable is if we separate:

- **pure-logic resolver tests** (exhaustive across dtype permutations), from
- **kernel correctness tests** (representative shapes), from
- **error-by-design tests** (stable error messages).

Phase 3 must add a minimal “complex smoke matrix” for the LinearAlgebra surface:

- `complex_float64`: add/sub/elementwise/matmul correctness vs NumPy
- `complex_float32`: same, smaller shapes + tolerances
- `complex_float16`: add/sub/elementwise/matmul correctness vs NumPy (two-plane storage) + persistence round-trip

## 9) Mandatory tests

These tests are required. They exist to prevent dtype coverage drift and to catch correctness/performance regressions early.

### 9.1 Pure-logic dtype resolution tests

Add unit tests (no kernels) that exercise the resolver tables/functions. At minimum:

- Fundamental kind rule: never promote down across `bit -> int -> float`.
- Float underpromotion: e.g., `matmul(float32, float64) -> float32` by default.
- Complex flag behavior: for each op, verify complex propagation/behavior is explicit (preserve/drop/error-by-design) and covered.
- Unsigned flag behavior: verify signed/unsigned mixing rules are explicit and tested.
- Error-by-design paths: verify they error with stable, specific messages.

These tests should be table-driven and exhaustive across the supported dtype set for each resolver entry.

### 9.2 Kernel/integration correctness tests

Add tests that validate numeric correctness and overflow behavior for representative ops and shapes:

- `dot`/`matmul` integer correctness across widths.
- Overflow throws deterministically (no silent wrap).

For reduction-aware accumulator widening specifically, add at least one test where:

- `matmul(bit, int16)` (or `dot(bit, int16)`) produces a value that would overflow an `int16` accumulator but fits in `int32` output.
- The test asserts:
  - correct numeric result,
  - accumulator-widen warning is emitted and mentions: op name, lhs/rhs dtypes, accumulator dtype, and output dtype.

### 9.3 Warning tests (user-facing behavior)

Add Python-level tests (and C++ tests where applicable) that validate warnings are:

- emitted when required,
- de-duplicated (warn-once policy),
- informative (message includes the dtypes involved and what is happening),
- suppressible/routable via a user-facing control.

Warnings to cover:

- float underpromotion warning (if enabled)
- integer overflow-risk preflight warning (heuristic)
- integer reduction accumulator-widen warning (deterministic)

### 9.4 Scale-first regression tests (bit materialization guard)

Add a regression test that guards the key scale-first property for `bit` operands:

- `bit` inputs must remain bit-packed during `dot`/`matmul` (no full materialization to an int/float element buffer).

Implementation note (testability): this may require a test-only hook (e.g., allocation tracer, “materialized_bit_elements” counter, or a debug trace flag) so the test can assert that no allocation proportional to `A.numel() * sizeof(int32)` occurred.

### 9.5 Support-matrix completeness test

The support matrix must be executable as a test/tool:

- It must fail CI if an op claims support for a dtype/structure/device combination that lacks an implementation or test coverage.

## 10) Acceptance criteria

- Adding a new operation requires changing:
  - the op implementation,
  - one promotion rule table,
  - one coverage declaration,
  - tests.
  It must not require “hunt across the codebase”.

- Complex dtypes are supported for float base dtypes only (`complex_float16/32/64`).

- Overflow behavior is consistent:
  - overflow throws,
  - large integer matmul emits a risk warning when appropriate,
  - no auto-promotion to avoid overflow.

## 11) Open questions (to confirm before implementation)

- Exact list of supported ops for “core coverage” in the support matrix (minimal set to enforce first).
- Whether unsigned + signed mixing rules should default to promoting to signed or throwing in ops that can go negative.
- Default behavior for numeric ops on `bit` when the semantic result is not representable in `bit` without widening: default widen vs error-by-design unless the caller explicitly requests an output dtype.
