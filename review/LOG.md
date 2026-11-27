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
- Closure matvec word-level: n=20000 top-6 PJ-like spectrum 1.8 s, 99 MB peak. eigvals_skew via zheevd (1.8x).
- ASan+UBSan clean after closures/bit kernels/zheevd (engine suite 2,899 passed, 0 reports).

## 2026-11-25 session 6
- Investigation: arbitrary-width integer dtypes (features/BIT_PLANES.md). No new user dtypes;
  F-055 (exact int matmul through float64 BLAS, 50x) is the bigger lever; bit planes later as
  an internal encoding behind value bounds. F-054 bit ops word path restored.
- Maintainer decisions: zeros/ones/empty default float64 with `dtype=` (F-041b done); tier-A
  last-digit float differences accepted for laziness.
- F-047/F-056 (eigen precision, float32 vectors), F-057 (RAM storage = anonymous memory;
  element-wise 3x faster than NumPy here), gemv matvec (23.6 -> 1.6 ms).

## 2026-11-25 session 7
- F-012: thread-pool spinning (OpenMP + our OpenBLAS + NumPy's OpenBLAS) was the dominant
  slowdown; `import pycauset` sets OMP_WAIT_POLICY / OPENBLAS_THREAD_TIMEOUT defaults.
  Export 250 -> 0.56 ms; dot via BLAS. r2_parity: most ops 0.8-1.1x (determinant 0.68x open).
  Benchmark no longer times an export inside every op.
- Warnings: removed the C++ warn-once set (it overrode Python's filter: simplefilter("always")
  could not re-enable a warning); removed accumulator-detail warnings (noise, no surprise).
- F-051: integer dot is an exact Python int (`vector_dot`); `v @ u` for int32/int16 had ignored
  the vector's factor (wrong result).
- Found: vectors lack `/`, `v - s`, `s - v`, unary `-`, `abs`, element-wise `*` and `/`; vector
  scalar ops turn int8/int64/uint* into float64 through double. Next: route vector ops through
  the matrix machinery (a dense vector is an n x 1 matrix over the same storage).
- F-058: vectors = 1 x n matrices over the same storage (`math/VectorOps.hpp`); vector
  arithmetic complete and NumPy-typed; matrix operators one binding path (`matrix_arith`,
  adds the missing `s / A`); ~1,400 lines of vector kernels/overloads deleted. Scalar kernels:
  contiguous same-dtype fast path + flat-block parallelism (`2*v` 6.4 -> 1.2 ms).
- Build note: ScalarOps.cpp instantiations are memory-hungry; the contiguous fast path is
  limited to same-dtype results (all-pairs OOMed a 15 GB container at -j1).
- F-051 closed: `pc.sum` (`math/Reductions.cpp`) returns NumPy kinds (exact int / float /
  complex), deterministic float order, popcount for bit storage (|C| n=20,000 in 2 ms), O(1)
  for closures; old device/solver sum (complex, OpenMP reduction, virtual per element) deleted.
- Linear algebra × views probe found F-059..F-063 (complex det/trace/inverse lost imaginary
  parts; matmul ignored conjugation incl. A.H @ B; eigvals ignored scale views; zero-offset
  windows read as contiguous in export/solve/svd; triangular det 0 for most dtypes). Fixed at
  the root: `is_subview()` replaces `has_view_offset()`, `write_out`/`is_plain`, LA entry
  guards. test_linalg_views.py (287 cases).
- Maintainer: the recurring "op X on dtype/structure/view Y is wrong or slow" class must be
  catalogued exhaustively, not found by accident: PLAN Phase C (operation atlas).
- Build note: the normal build OOMs at -j2 now (CpuSolver + ScalarOps); use -j1.

## 2026-11-25 session 8
- Atlas C.1 done, C.2 first run: 60,738 cells. Wrong cells traced to 12 engine bugs
  (F-064..F-075, all fixed and verified by direct reproduction) plus oracle bugs (integer
  cast overflow, np.conj(bool) -> int8, complex inputs cast to real for norm, 6-digit sort
  for spectra, signed zeros). Engine changes: triangular clone keeps has_diagonal;
  antisymmetric needs a signed dtype; triangular matmul orientation/diagonal; native solve
  for every dtype/structure + vectors (?gesv), solve_triangular/lstsq/slogdet rebuilt on it;
  device `determinant` replaced by `lu_pivots`; conj-view complex factor; window save; ~ on
  structures; mixed int kernel dense-only. Python: matmul/solve/matrix_power shortcuts and
  silent fallbacks removed (F-067, F-068, F-073).
- Maintainer asked why so many simple bugs exist. Answer recorded as Phase N in PLAN: four
  patterns (dtype-selected kernels that assume layout; optional view flags; duplicated rules;
  silent fallbacks) plus example-based tests that never enumerated combinations.
- Decisions taken (maintainer may veto): `~` follows NumPy (complements every element of a
  structured matrix); user-set properties no longer steer dispatch or change results (old
  tests asserting that removed: test_properties_dispatch.py, two others).
- Maintainer approved Phase N and stated the review's purpose: the code was grown feature by
  feature (AI-generated over ~a year) without integration; integrating it is part of the review.
- Atlas reruns: 2,446 wrong cells → 101 → 6 → 0 (F-076..F-079 fixed on the way; oracle made
  deterministic: CRC32 seeds instead of per-process str hash, well-conditioned operands for
  linear algebra, float16 overflow compared as float16). CI test passes (40 ops, 34 s).
  Remaining report: 8,942 generic and 3,056 NumPy-round-trip cells, 1,399 unsupported → Phase N.
- Phase N.1 started with matmul (F-080). The device kernels now state their contracts and
  throw on anything else; `dispatch_matmul` is five rules plus one normaliser. Removed: ~25
  front branches, 9 per-element device loops, 4 member `multiply` entry points (net -930
  lines). Atlas: matmul generic 519 -> 0, total generic 8,942 -> 6,386; still 0 wrong cells.
  Classifier: on O(n³) ops up to 4n² accessor calls are normalisation (documented).
  Old test `test_bit_mixed_fastpaths.py` now asserts the one `cpu.matmul.bit_x_dense` tag.
- Maintainer asked: does Phase N scale to new views/structures/ops; why no float16 diagonal;
  is "fast" NumPy speed. Answers: a shared converter (N.2) + declarative normaliser (N.1b)
  make a new view/structure one change; the structure gaps have no obstacle (N.5, one
  decision on complex symmetric vs Hermitian); "fast" was only a path label, so the timing
  pass (C.3) was built first.
- Timing pass findings: F-081 (cached sum/det/trace/norm/eigen/inverse went stale on
  `M[i,j] = v`; trace/det answered from user flags; file-backed inverse forced to NumPy):
  all result caching removed (~1,000 lines, 2 modules, 2 test files). F-083: per-call
  overhead 30-50 µs; virtual-call conversions dominate cubic kernels; the interim
  "4n² is normalisation" classifier rule reverted (matmul honest count: 557 fast, 290 generic).
- N.2 converter (F-084): one written view rule + typed readers per class in its own TU (the
  first attempt instantiated one reader per result dtype x class and OOM-killed the compiler;
  narrowing through int64/float64/complex128 fixed it). First version recursed for complex
  results (stack overflow) and gave +0.0 for 0.0 x (-1-0j): both caught at once by the
  atlas. Generic 8,598 -> 4,986, matmul 837/847 fast. Atlas window no longer includes the
  export of the result (it double-charged export reads to the op).
- Maintainer: the operation procedure must be documented and survive; the missing types are
  a big hole ("the heart of PyCauset"); complex symmetric and Hermitian are separate. Done:
  `Adding Operations.md` rewritten (Phase N procedure), `Adding Types.md` rewritten (the
  catalogue), `CLAUDE.md` entry point. N.5.1-N.5.4: one catalogue (F-087), 9 new structure x
  dtype pairs + Hermitian/skew-Hermitian; the new pairs found 5 existing bugs (45 atlas ✗),
  all fixed at the root. Atlas 77,372 cells, 0 ✗.

- N.3 (F-090, F-091): every linear-algebra and vector-product entry in the engine.
  `CpuLinalg.cpp` templated LAPACK kernels (eigh, eig, cholesky, qr, lu) for the four
  LAPACK dtypes; native pinv, qr modes, outer (any lengths), cross, norm(ord), matrix_rank,
  cond. `_internal/ops.py` makes no NumPy computation. Found on the way: matrix_rank of a
  triangular matrix counted its diagonal (every causal matrix had rank 0); four property
  shortcuts changed results; file-backed operands sent matmul/invert through a per-element
  NumPy path (float results). New atlas ops eig, eigh (residual checks; eigh against the
  Hermitian matrix of the lower triangle, as LAPACK reads it). Atlas 79,781 cells: 0 ✗,
  NumPy round trips 4,209 -> 0, ok 39,123 -> 41,570 (lu's 908 raises gone).
- Properties removed (maintainer: flags, then user notes too; F-092) with their C++ bitmask,
  GPU-routing hook, wrappers and saved-file namespace; F-082 helpers return structured classes
  (`pack_as`); `pc.hermitian`. F-093: loaded files were written in place; now copy-on-write
  mappings and atomic saves. F-094: typed row/diagonal readers for every class; det/slogdet
  (structure rules + one ?getrf template in NumPy's precision), sum, trace, bool write-out,
  one NumPy exporter, element-wise rows. Atlas harness: single-precision linear algebra is
  checked against NumPy in single precision with single-precision tolerances (the tolerance
  came from det's float64 result dtype). Atlas: 0 ✗, generic 10,152 -> 3,730, fast 31,379 ->
  37,599. N.6 first cut: debug strings were built on every allocation (~28 µs per op).
- F-095: matvec/vecmat are matmul with the vector as a column/row (~1,160 lines of kernels
  and device interface deleted; the bit-closure kernel kept as a structure rule). Moving them
  surfaced F-097: hand-built view copies (multiply_scalar / transpose per class) passed the
  transposed shape as the storage shape and swapped window offsets; the square-only atlas
  never saw it. Now two shared O(1) views (scaled_clone, transposed_clone) and atlas views
  rect_T / rect_T_scaled. F-096: per-call overhead first pass (debug strings, trace
  deep-copy, pool wake-ups for tiny loops): pc.sum 51 -> 0.7 µs, pc.matmul 136 -> 54 µs.
  Atlas 88,361 cells, 0 ✗, fast 40,212, generic 1,556 (from 10,152 this morning).
- N.6: stack sampling (gdb) found the MemoryGovernor re-reading /proc/meminfo on every
  allocation; with that, heap storage for small objects and the earlier cuts, `a + b` on
  4 x 4 went 29 -> 1.9 µs. F-098: export's silent Python fallback removed; dot, diagonal
  matmul, inverse rules and complex32 element-wise typed. Atlas: 88,361 cells, 0 ✗, fast
  41,529, generic 239 (10,152 at the start of the night).
- Timing pass with pycauset imported first and paired sampling (11,292 cells, n = 1024 for
  O(n^2), 384 for O(n^3)): linear algebra is near NumPy (dense float64: det 0.97x, slogdet
  0.97x, solve 1.7x, svdvals 1.4x, eig 0.75x, eigvals 0.74x, inv 0.82x; medians over every
  structure: eigh 0.81, cholesky 0.75); element-wise O(n^2) is the gap (dense float64 add
  0.81x, add_scalar 0.33x, complex-factor scaling 0.21x; structured element-wise ~0.1x;
  per-call-bound ops such as trace and dot at 0.05-0.2x). The 0.9x target is met for 2,847
  of 11,292 cells. Found on the way: two OpenBLAS pools (NumPy's and ours) spin against
  each other when NumPy is imported first (warning + import order; one-pool build = PLAN 6.0);
  fresh 8 MB results page-fault on every write (calloc storage, in build).

## Session 9 (2026-11-27): order changed to cleanup first
- Maintainer asked whether cleanup should precede features. Agreed: the safety net (atlas,
  suites) exists now, and speed/features built on code about to be split or deleted would be
  done twice. PLAN order: Phase 4 cleanup -> test health -> docs truth -> speed and features.
- F-099: packed-storage element-wise rule (sym+sym 1.5x NumPy, diag+diag 21x); tri+tri needs
  `has_diagonal` in the resolved result shape (open, with 4.3).
- 4.1: dead-code tools re-run; ~900 lines of dead C++/Python removed (helpers, uncalled
  device methods and their CPU/AutoSolver/CUDA implementations).
- 4.7 / F-100: the IO trace reported a Python routing guess that the engine never read;
  `set_io_streaming_threshold` controlled nothing (a test that "forced streaming" with it never
  streamed). Planner deleted; traces now carry operand facts and the engine's kernel log.
  Gates: 4,448 + 3,735 passed, atlas 88,361 cells 0 wrong, CI gate 42 passed.
- Test retirement (4.5, TESTS.md §4): per-file Python coverage plus a C++ coverage build
  (`build/cov`) comparison. 40 old files deleted once their unique lines were covered by three
  new files (warnings, API surface, interop); 19 physics files moved to `tests/physics/`.
  The new interop tests found and fixed F-102 (matrix + vector), F-104 (`m[...]` broken by
  a null `py::slice()`, list / NumPy-integer indices, `ndarray @ matrix`); the five matmul
  bindings are one helper and `.multiply` is gone. F-103 (Python int -> int32 vs NumPy int64)
  is open for the maintainer. Gates: 4,443 + 3,923 passed, atlas 0 wrong, CI gate 42.
- Maintainer decisions: F-103, Python integers stay int32 (documented rule; no value-based
  narrowing in constructors); order stays cleanup -> tests -> docs -> speed -> features, F1
  (value bounds) included, which later makes minimal-dtype storage safe.
- 4.5 done: `tests/python/` retired (72 files deleted after coverage checks, ~8,100 lines; 19
  physics files -> tests/physics, 9 block-matrix files -> tests/blockmatrix). New engine
  files: test_storage, test_persistence (one container header reader now, the duplicate
  removed; tie-break documented), test_stress (fuzz, threads, leaks: coverage can't see
  them; threaded I/O verified on Linux, 20/20). F-106: OpRegistry removed. storage_debug.py
  (unused) removed.
- Physics parked (maintainer): `experimental/physics/` holds causet/spacetime/field/
  synthetic/vis, the native sprinkler and spacetimes, their bindings, tests, docs and
  guides, and the gallery script (README with a bring-back checklist). The link-matrix
  kernel moved into the engine (`src/math/CausalMatrix.cpp`; windowed views rejected,
  portable `countr_zero`). Docs pass started with the pages physics touched: README,
  Quickstart, Examples, the three index pages; their code runs in `test_doc_snippets.py`
  (which caught `pc.inverse` in my own Quickstart draft).
- Maintainer: causal sets are R1. Back in the package: CausalSet / causet, spacetime
  (MinkowskiDiamond only), synthetic, the native sprinkler. F-105 fixed (true d-dimensional
  diamond, one native implementation), F-107 fixed (validation path counts). Fields, plots,
  other spacetimes stay parked. No backwards-compatibility shims (no users). Causal-set docs
  and guides back, edited to R1 and run in test_doc_snippets.py.
- Maintainer: lightweight plotting is R1 (a later release is about visualization). Back in
  the package: `pc.vis` (embedding, Hasse diagram, causal matrix, `show`), the CausalSet plot
  methods; `MinkowskiDiamond` declares `boundary()` / `display_axes()`. Tests in
  tests/engine/test_plots*.py (cylinder / box / custom-spacetime cases dropped); the
  Visualization guide and vis pages edited to the diamond and run in test_doc_snippets.py
  (`fig.show()` stubbed).
- `plot_hasse` drew links from a NumPy `C & ~(C @ C)` product; it now uses the engine's
  transitive reduction (`CausalSet.links()`, or of the induced suborder when subsampled).
  The unused `to_embedding` hook is gone from vis.py.
- Docs truth pass (Phase 5): Compute Architecture rewritten (CPU-only path, kernel file map,
  LAPACK routines per operation, dormant GPU hooks named); Algorithms rewritten (no GPU, the
  real sprinkler: it holds all coordinates while it runs, serial O(n²) pair loop, on the
  Phase 6 list); Memory and Data direct-path rule (no pinning tiers); Block Matrices leaf
  dispatch (by type, no fallback); NumPy Integration (`copy=False` never gave a view:
  export is always one copy). Next: documentation/dev and documentation/project.
- Docs truth pass, documentation/dev: Codebase Structure, Bindings & Dispatch, Python
  Internals, Testing & Benchmarks, Build System, Square-only Assumptions rewritten to the
  code; Warnings & Exceptions gains "the engine never prints"; Restructure Plan,
  performance_optimizations, R2_QA_AUDIT, R2_PERF_FINDINGS deleted (history, wrong now).
  BENCHMARKS.md rewritten: how to measure, today's numbers (LAPACK ops 0.5-0.7x, F-108),
  where PyCauset is ahead by construction; the stale Windows-era graphs deleted.
- Scripts: benchmarks/bench_coverage and bench_structures (superseded by the atlas and its
  timing; bench_structures used the removed properties), tools/stress_100gb_matrix
  (FloatMatrix, a maintainer-local path), tools/macos_hang_diagnose (its bug is fixed),
  tools/check_native_exports (a list of class names removed long ago),
  tools/audit_public_surface (now a test) and tests/stability (FloatMatrix; test_stress
  covers it) deleted. bench_ram fixed. The continuum-limit benchmark and the field /
  spacetime demos moved to experimental/physics/scripts, install_cuda12.ps1 to
  experimental/cuda/tools. Empty tests/physics removed.
- F-109 (engine prints; header mapped as data) and F-110 (internal native helpers public;
  dead `pc.seed`) fixed. `_debug_resolve_promotion` is the native function (the Python
  wrapper only re-checked that it exists).
- C++ tests: `tests/test_symmetric_matrix.cpp` no longer compiled (old SymmetricMatrix
  constructor and AntiSymmetricMatrix class; an assert-based main linked with gtest_main),
  so the CI C++ job could not build. Deleted; its one uncovered behaviour (antisymmetric
  diagonal writes raise, lower writes store the negation) is `test_antisymmetric_storage_writes`.
  ctest: 5/5 suites pass. Stale binaries of removed tests (op registry, lazy) cleared.
- Gates: 4,111 + 4,053 (lazy off) passed, atlas CI 42, ctest 5/5.
