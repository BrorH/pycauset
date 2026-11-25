# Work Log

One entry per working session, newest at the bottom. Say what you did, what you
found, and what you left half-done, so the next agent can continue without
re-deriving anything. Keep entries factual and short. Details belong in
FINDINGS / ARCHITECTURE / area notes; link to them.

---

## Session 1: 2026-11-24, kickoff (Claude Code, branch `ccr-366933fe-71cbew`)

**Brief from the maintainer:** start a thorough review and health check of the
whole repository: catalog everything, plan the multi-week overhaul, document it in a
dedicated directory for future agents, test heavily, and propose cleanup and
restructuring. Pre-alpha, so no backwards compatibility. GPU work is on hold. Mid-session the
maintainer asked for an overview and plan *before* deep testing, and for more
verbose progress reporting. The documents below were written after that steer.

**Done**
1. Built from source (pip/scikit-build, plus standalone CMake with `BUILD_TESTS`,
   plus an ASan+UBSan build). Environment gotchas recorded in BASELINE.md.
2. Baseline: Python suite 799 passed / 34 skipped. 6/6 wired gtests pass. Lint and
   warning counts recorded.
3. Un-skipped the three "crash/segfault/unverified" tests: all pass (F-009).
4. Built the three orphaned gtests: `test_parallel_utils` hangs → **F-004** (nested
   ParallelFor deadlock). Reading the code also gave F-005.
5. Wrote four probes (`review/probes/`):
   - NumPy-fallback detector → eigh/eigvalsh/svdvals/matrix_rank always use NumPy
     (F-011); `cond` uses the wrong default norm (F-003).
   - Out-of-core consistency → core ops correct when forced to disk.
   - dtype × op grid → **F-001** (int32 `A*3` is a silent no-op), F-002 (no `-A`,
     `3*A` fails, ...), F-013 (bool semantics).
   - Doc-snippet runner → F-008 (≈20 broken examples).
6. ASan+UBSan run (first attempt, UB fatal): stopped at the first UB, signed overflow
   in the int32 add kernel → **F-006**. A second build with UB set to recoverable was
   started to get a full pass. See "Left open".
7. Stray `_vals`/`_vecs` files in the repo root led to **F-024**: every `eig`
   result is backed by the same `./_vals` file, so a second `pc.eig()` silently
   corrupts the first. It was masked by `.gitignore` entries.
8. Read the architecture end to end and wrote ARCHITECTURE.md. Wrote INVENTORY.md
   (154 source files with a review column), FINDINGS.md (F-001…F-024), PLAN.md
   (phases 0–6), RESTRUCTURE.md (proposal plus 8 open decisions), BASELINE.md.

**Incidents**
- The first uncapped doc-snippet run filled the disk (a doc example allocates a
  ~125 GB causal set; see F-008). That killed the parallel ASan build. Fixed by
  capping file size in the runner and deleting the working dirs.

**Left open / next steps**
- Maintainer to answer RESTRUCTURE.md §3 decisions.
- Finish the full ASan+UBSan pass and log the results (BASELINE row, new F-ids).
- Start PLAN Phase 1 (safety net). Suggested first PRs: 1.1 test harness/conftest,
  1.2 wire gtests into CI, 1.4 remove stale skips.
- Not yet done: coverage measurement (1.9); checking whether any production
  code path actually nests ParallelFor (affects F-004 severity).

## Session 1, part 2: 2026-11-24, new direction from the maintainer

**Steer:** (1) design a new, compact, smarter test suite instead of keeping the
16.7k-line one; (2) the maintainer is creative director and the agent is technical
lead, so make technical calls and ask only plain-language product questions;
(3) make dead-code elimination a first-class part of the overhaul, and look for
existing tools; (4) R1 = the engine; physics is deprioritized.

**Done**
1. Full ASan+UBSan pass: no memory errors; UB in int32 add/mul (F-006) and a null
   memcpy (F-025). Learned that pytest capture hides UBSan output (F-027).
2. Dead-code tooling researched and trialled: Skylos (best static Python), Vulture
   (noisy), cppcheck (useful only intersected with coverage), coverage.py with
   per-test contexts, and gcov/gcovr. Wrote `scripts/coverage_analysis.py` and
   `scripts/cpp_unexecuted.py`; snapshots in `review/deadcode/`. Wrote DEADCODE.md.
3. Coverage results: Python 76% (87 never-executed functions); C++ 35% lines /
   489 of 1,562 functions never executed; **StreamingManager.cpp 0%** (F-026); a
   pure-Python fallback matrix still wired into `pc.matrix` but never reached.
4. Test redundancy: 47/121 test files contribute zero unique Python lines. Wrote
   TESTS.md (oracle-first, table-driven, invariant-based; ~3–4k lines target;
   migrate, then retire old files only with coverage proof).
5. RESTRUCTURE §3 rewritten: 9 technical decisions made (vetoable), 2 product
   questions left open. PLAN updated: R1 engine priority, Phase 1 = new suite core,
   new Phase 1b = Tier-A dead code, physics review moved post-R1.

**Next:** Phase 1 (new-suite core, CI wiring) and Phase 1b (Tier-A removals), in
parallel. The C++ coverage run should be repeated with the gtests included before
removing MemoryGovernor's pinning API.

## Session 1, part 3: 2026-11-24, maintainer answers the decisions

- Confirmed: operators in C++ only (Python stays a thin packaging layer: "the module
  is Python, the engine is C++"); CUDA → `experimental/`; renames; generated API
  reference, one page per function like NumPy, short and slop-free; export guard by
  size only.
- Delegated to the tech lead: properties ownership (→ C++); dtype scope (→ NumPy's
  numeric set + bit; drop complex_float16; add none).
- Reversed: spill dir stays `./.pycauset` (maintainer wants full control over file
  lifetime). F-017 → wontfix.
- bit/bool addition = integer counting (F-013 decided).
- **New feature accepted:** value bounds and bounds-aware dtype selection.
  Spec: `features/VALUE_BOUNDS.md`; PLAN Phase F1. It replaces the existing
  heuristic int32 matmul overflow preflight (`bind_matrix.cpp:231-305`), which
  samples a few elements and is therefore unsound.

## Session 1, part 4: 2026-11-24, R1 scope answers

- **All 16 dtypes stay** (maintainer: small dtypes are the under-promotion mantra).
  Reverses the tech lead's "drop complex_float16". int64 is the ceiling.
- Keep the skew-symmetric eigensolvers and Arnoldi: their untested C++ paths become
  must-test items in Phase 2.
- `precision_mode` (+ set/get) deferred out of R1: to be removed (PLAN 4.1b).
- bool/bit addition counts into integers, documented (reconfirmed).

## Session 1, part 5: 2026-11-24, value-bounds refinements (maintainer)

- Maintainer's framing: track the "highest-resolution element" through chains of
  ops, with rules for every op, cheap per-element writes, and "unknown" as an
  allowed state. Working names: `value_bounds` internally, `A.minimal_dtype` user-facing
  (final name is the maintainer's call).
- Tech-lead decisions added to the spec: guaranteed bounds vs observed values are
  kept separate; **no sampling for dtype choice** (unsound); a **predict-then-measure**
  design (rules choose the dtype before the op; the kernel records exact min/max while
  writing), so bounds don't loosen over long chains; out-of-scope list.

## Session 1, part 6: 2026-11-24, execution begins (maintainer: "start executing")

**Policy added:** deterministic only, no sampling/guessing in R1 (README rule 12).

**Commits / changes**
1. `fix(F-028)`: removed the sampled int32 matmul overflow preflight (the exact
   int64-accumulator overflow check remains). Full guesswork audit in FINDINGS F-028;
   float reductions verified bit-identical across runs.
2. New suite started: `tests/conftest.py`, `tests/engine/` (isolated storage and cwd),
   `test_regressions.py` (F-024 and F-028 tests; strict-xfail for open F-001/2/3).
3. **F-029 found and fixed:** the old suite imported a stale site-packages copy, so
   local runs weren't testing the source tree. pyproject now sets `pythonpath = ["python"]`.
4. `fix(F-024)`: eig/eig_skew results get unique backing files; `.gitignore` masking removed.
5. Tier-A dead code batch 1 (Python + C++), see DEADCODE.md: lazy_ufunc.py,
   interop.py, WorkloadPartitioner, main.cpp, unused AutoSolver/CpuSolver/SystemUtils/
   binding_warnings functions, test-only ObjectFactory load_* / flush_async /
   promote_to_ram (and their test cases), unused Python helpers, unreachable code.
   CMake leftovers, `#pragma message`, `_GNU_SOURCE` redefinition, all 15 `src/`
   compiler warnings (incl. a zombie "Send Memory Hints" block that sent nothing).
   Wired 2 orphaned gtests (ctest 8/8).
6. New findings: F-030 (eigvals_arnoldi silently uses dense NumPy; the native solver
   reportedly crashes), F-031 (a C++ test file asserts nothing in release builds).

**State:** old+new Python suites 802 passed / 34 skipped / 9 xfailed; ctest 8/8;
0 compiler warnings in `src/`.

## Session 1, part 7: 2026-11-24, critical fixes

- `fix(F-004,F-005)`: ParallelFor no longer deadlocks when nested (runs inline on pool
  workers) and waits for all tasks before rethrowing. Atomic thread count. New gtests
  verified to hang/fail on the old code. ctest 9/9.
- `fix(F-006,F-025)`: integer + - * wrap without UB (one set of wrapping functors in
  ScalarUtils.hpp); null memcpy guarded. **ASan+UBSan: 0 errors, 0 UB** (811 tests).
- F-001 root-caused; design decided and written into FINDINGS (integers never carry a
  lazy scalar; typed kernel with NumPy result dtypes). Implementation deferred to the
  next session because it belongs with the C++ operator-table consolidation (PLAN 4.3).
- **Benchmark correction:** the first parity baseline was taken under build load and is
  invalid. Quiet-machine numbers: only 2/8 ops at ≥0.90× NumPy; dot is ~100× slower
  (F-012). BENCHMARKS.md's "8/8" claim must be re-measured on the maintainer's machine.

**State at end of session 1:** Python 811 passed / 34 skipped / 9 strict-xfail (open bugs
F-001, F-002, F-003); ctest 9/9; 0 compiler warnings in src/; sanitizers clean.

**Next session, suggested order:**
1. F-001 (typed integer scalar kernel), then F-002 (`-A`, `int * A`, `A - int`) and
   F-003 (`cond`), which flips the 9 strict-xfails.
2. F-030: native Arnoldi crash (needed for R1); then skew eigensolver tests.
3. TESTS.md core: `tests/spec.py` and the conformance grid over all 16 dtypes.
4. CI: build and run the gtests; a UBSan job with `-s`; ruff-F and Skylos gates.
5. F-026: decide whether the C++ StreamingManager lives or dies (real larger-than-RAM run).

## Session 1, part 8: 2026-11-24, policy clarification: guesses may inform, never decide

- Maintainer clarified: guesswork is fine for warnings, advice and explicitly
  statistical operations; it must never change results (values, dtype, success/failure).
- The sampled int32 matmul overflow preflight (removed in `fix(F-028)`) was therefore
  allowed after all, and useful (an early warning before a long run ends in
  OverflowError). **Restored**, labelled advisory in code and message, garbled "≈"
  fixed, its test restored. F-028 tests now assert exactness whether or not it fires.
- README rule 12, FINDINGS F-028, VALUE_BOUNDS spec, and the warnings/dtype developer
  docs updated to the clarified rule.

## Session 2: 2026-11-24, scalar operators, export, cond, Arnoldi diagnosis

- **ScalarOps** (`include/pycauset/math/ScalarOps.hpp`, `src/math/ScalarOps.cpp`): one C++
  entry point for `A ⊕ s` (+ − * / and reversed, `-A`) with NumPy 2 result-dtype rules
  (weak Python scalars, strong NumPy scalars), exact typed kernels for integer/bit
  matrices, and triangular structure kept when 0 ↦ 0. Bindings: early exits in 17 typed
  overloads, 7 catch-alls now understand numbers (the pybind overload-order bug behind
  `A - 1` / `3 * A`), strict `py::int_` instead of `int64_t` (NumPy ints were losing their
  strong dtype via `__index__`). Fixes F-001 (wider than recorded: every int AND bit dtype,
  incl. causal matrices), F-002.
- **F-003:** `cond` now follows NumPy for every `p`.
- **F-033:** native NumPy export existed for only 3 of 13 dense dtypes; replaced ~150
  duplicated lines with one generic exporter; the Python fallback reads ints exactly.
- **F-032** found: unregistered template instantiations reach Python as bare `MatrixBase`.
- Two old tests that **encoded bugs** were corrected (int32 × 3.5 "should remain
  IntegerMatrix"; cond "should be Frobenius").
- **F-030 diagnosed:** the native Arnoldi no longer crashes, but it's an unrestarted
  single pass that ignores `tol` (silently inaccurate, ~1e-3). Plan: restarted method with
  convergence check (Spectra or Krylov–Schur). Public function stays on the exact NumPy path.
- New test file `tests/engine/test_scalar_ops.py`: 566 cases vs NumPy (values + dtypes).
- State: 1,398 passed / 34 skipped / 0 xfail; ctest 9/9; ASan+UBSan clean; the dtype probe
  is all "ok" except the documented bool counting policy.

## Session 2, continued: CI, F-034, conformance grid

- CI: `cpp-tests` job; ASan+UBSan (fatal) on PRs; both suites run (not yet observed on
  GitHub: workflows trigger on main/PRs only).
- Dtype-pair survey (980 combos) → **F-034 fixed**: int ⊕ float could overflow to inf
  (int32 100000 + float16 = inf) and int32 / int32 was float32; cross-kind results now hold
  the integer dtype; within-float under-promotion unchanged. **F-035 recorded**: mixed
  matmul with float16/complex has no kernel (64 combos); signed int ⊕ uint64 raises (32).
- `tests/engine/test_conformance.py`: every dtype pair × 5 ops; 894 pass, 104 xfail (F-035).
  Two of my own test-reference mistakes were fixed along the way (unsigned wraparound,
  complex16 export name); the library was right in both cases.
- F-032 (unregistered instantiations) scheduled with the rename sweep (PLAN 4.6).
- State: 2,295 passed / 34 skipped / 104 xfailed in ~20 s; ctest 9/9.
- `fix(F-035a)`: mixed-dtype matmul with float16/complex promotes operands via new
  `pycauset::cast_to` (72 grid cases now pass). F-036: the identity-matmul shortcut
  bypassed dtype rules and F-001's invariant; fixed, and `matrix_power` no longer seeds
  with a float64 identity. F-035b (signed ⊕ uint64) keeps raising by decision until value
  bounds. Gap cases in the grid now *run* under xfail so fixes surface as XPASS.
- State: 2,368 passed / 33 skipped / 32 xfailed (F-035b); ctest 9/9; ASan+UBSan clean.
- `tests/engine/test_invariants.py`: 400 cases (catalog of dtypes, views, causal powers,
  structured types × RAM/disk): element access, export, sum, transpose, @I, save/load agree.
  It found **F-037** (save/load of transposed views returned the untransposed matrix:
  silent corruption), fixed, and **F-038** (LazyMatrix lacks sum/.T/save), open.
- State: 2,760 passed / 33 skipped / 44 xfailed (F-035b, F-038).

## Session 3: 2026-11-24

- **precision_mode removed end to end** (R1 scope): C++ mode enum, thread-local and scoped
  guard; `set/get_precision_mode` bindings; `pc.precision_mode`; the `precision_mode` argument
  of `_debug_resolve_promotion`; 3 tests; 3 API pages, nav and index entries; guide
  sections. The under-promotion warning now says "Cast both inputs to float64". Also fixed a
  guide that claimed float32+float64 → float64 (it's float32, documented under-promotion).
- **F-030 fixed:** `eigvals_arnoldi` rewritten on Spectra (new dependency, header-only,
  MPL-2.0, system or fetched; THIRD_PARTY_NOTICES updated). Symmetric → Lanczos (float64),
  general → Arnoldi (complex128), raises on non-convergence, deterministic, works on
  disk-backed matrices via our matvec. Public function routed to native; NumPy fallback,
  stale crash comment and the Python streaming variant deleted. Accuracy 1e-3 → 1e-14.
  New `tests/engine/test_eigensolvers.py` (14 cases incl. skew solvers on causal matrices).
  API page rewritten (example verified by the snippet runner).
- Note: a "must raise on non-convergence" test proved unreliable (Spectra's residual
  estimate can underflow to 0 after many restarts), so it was replaced by a clustered-
  spectrum accuracy test. The raise path is covered by code review only.
- **F-038/F-039:** measured lazy vs eager (no benefit) → operators eager; found and fixed
  `C + C.T` / `C - C.T` silently dropping the lower half (two inconsistent structure
  resolvers, neither orientation-aware; now one). Then **deleted the expression engine**:
  `include/pycauset/matrix/expression/` (8 headers), `MatrixOps.{hpp,cpp}`,
  `bind_expression.cpp`, `MatrixBase`'s expression-based assignment operators (the compiler
  confirmed no other users; they also copied through a double buffer, which drops imaginary
  parts), 2 C++ lazy test files, the lazy-only Python test file, `LazyMatrix` names in
  Python, the internals page and lazy claims in 4 guides. The NumPy-ufunc section of the
  NumPy guide (false since before; F-008) now states the truth.
- State: 2,787 passed / 32 xfailed (F-035b); ctest 7/7 (2 lazy test programs removed).
- **F-026 resolved / F-040 fixed:** the C++ streaming engine is live but was unreachable
  in tests (it keys off real free RAM, not `set_memory_threshold`). Added internal
  safety-margin helpers and a trace to force and verify it: matmul and element-wise
  streaming are correct for plain operands, but views (transpose, slices, scalar views)
  silently produced wrong results and complex/mixed dtypes threw. Streaming is now
  restricted to what it handles; everything else uses the direct kernels.
  `tests/engine/test_streaming.py` (54 cases).
- Sanitizer builds now use `-j1`: two parallel sanitized compiles of the monolith files
  exceed the container's memory (OOM-killed twice). Another argument for splitting them.
- F-041a: `pc.TriangularMatrix` was a constructible pure-Python toy matrix; now an abstract
  isinstance type (its docs already said so); ~60 dead lines removed. F-041b (dtype-deferred
  `zeros()` wrapper) left for the maintainer: it was a deliberate R1 product feature.

## 2026-11-24 session 4
- Maintainer: lazy evaluation is a wanted feature for ALL dtypes and structures; plan it thoroughly.
- Probe `review/probes/view_semantics_probe.py`: views exist for float scale and all transposes; integer/bit/vector scaling copies (F-042); no mutation leaks; Diagonal/Symmetric .T copies (F-044).
- Measured: 2A+3B costs 2x A+B (F-043); (A@B)@v 236 ms vs A@(B@v) 41 ms at n=3000.
- Wrote features/LAZY.md (exactness tiers E/A, catalogue L1-L6, architecture ViewSpec/ExprNode, test oracle, phases). PLAN Phase L added.
- CUDA moved to experimental/cuda (dormant core hooks listed in its README).
- MatrixType -> Structure enum (causal = TRIANGULAR + BIT).
- One Python class per structure with .dtype (RESTRUCTURE 3.1 item 4 revised): DTypeDispatch.hpp
  visitor, polymorphic type hook, generic element access; Python facade, persistence, tests and
  docs migrated. Findings: F-045 (view get/set wrong), F-046 (size-based float32 + structure
  inference), F-047 (eig float32 -> float64, open), F-048 (stale installed package in tests/probe).
- Suite: 2,825 passed / 7 skipped / 33 xfailed; ctest 7/7; doc pages failing 70 -> 60 (F-008).

## 2026-11-24 session 5
- Causal-set lazy forms checked (n=600): K strictly upper triangular; W = (sqrt(AᵀA)+iA)/2 holds (rel 6e-8 via the squared route); real eigh 22 ms vs complex 87 ms; sum(C@C) = (1ᵀC)(C1) exactly. Added LAZY.md L7.
- F-043 fixed for element-wise ops (StridedReader); F-049 int64 view precision, F-050 conj/transpose/symmetric-clone view bugs fixed; tests/engine/test_views.py (326 cases). Suite 3,151 passed.
- L1: integer scale views (dense+triangular) on; readers: get/getitem/export/strided/matvec/fold/~ via materialize; 5*(3*(2*A)) int32 85 -> 0.21 ms.
- ASan+UBSan clean on the engine suite after the session-5 kernel/reader changes (2,830 passed). L2a done: A±Aᵀ packed.
- L2b: BitClosureMatrix (C±Cᵀ over bits), _set_lazy oracle, is_lazy, pc.materialize; test_closures.py (61 cases). 3,671 passed.
