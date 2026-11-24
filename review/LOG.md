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
