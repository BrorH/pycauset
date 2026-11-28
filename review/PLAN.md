# Codebase Review & Health Overhaul: Master Plan

**Goal:** every line of PyCauset reviewed, every claim in the docs true, the
repository cleanly organized, and the engine "smooth as butter": no silent wrong
answers, no hangs, NumPy-compatible where it claims to be, fast where it claims to
be. Expected duration: weeks. Many agents, and the maintainer, will work on it.

**Scope:** everything (C++ engine, bindings, Python layer, physics suite, tests,
benchmarks, tools, build and CI, docs). **Out of scope:** GPU/CUDA work (on hold;
moves to `experimental/`, RESTRUCTURE §3.1) and new features.

**Physics for R1 (maintainer, 2026-11-27):** causal sets are R1 (the tools to create and
test with): `CausalSet` / `pc.causet`, sprinkling into the d-dimensional Minkowski diamond,
the order methods (links, chains, layers, intervals, past/future), Myrheim-Meyer, synthetic
orders, save/load, and lightweight plots (`pc.vis`; a later release is about visualization).
Fields and the rest of the spacetime library are parked in `experimental/physics`. No backwards compatibility (no users yet).

**Priority for R1 (maintainer, 2026-11-24):** the *engine*, meaning linear
algebra and "NumPy for causal sets". The physics suite is **deprioritized**: smoke
tests only, and fix glaring bugs if found, but no deep review or new spacetimes
until after R1.

**Companion strategies:** `TESTS.md` (new compact test suite) and `DEADCODE.md`
(dead-code program). Both run *through* the phases below.

**Order from here (2026-11-27, maintainer prompt, tech-lead decision):** the safety net now
exists (atlas: 88k cells, 0 wrong; both suites green), so cleanup comes *before* further speed
work and before any feature:
1. Phase 4 cleanup: 4.1 dead code B–D, 4.2 split the giant files, 4.8 silent-exception sweep,
   4.3 the Python patch stack, 4.7 one StreamingManager.
2. Phase 1 test health and 4.5 (retire the old suite).
3. Phase 5 documentation truth pass.
4. Then Phase 6 speed (open gaps in FINDINGS F-096/F-099) and the features D, L, F1.
Speed and features built on code that is about to be split or deleted get done twice.
Maintainer confirmed this order (2026-11-27), including F1 (value bounds) after speed work:
F1 is also the long-term answer to F-103 (storage at the minimal dtype becomes safe once
each op predicts its result range).

**How to use this file:** phases run roughly in order, but within a phase the work
items are independent and can be claimed by different agents. Claim an item by
writing your session/agent name next to it; tick it when it's merged. Record every
session in `LOG.md`. Record every issue in `FINDINGS.md`.

---

## Phase 0: Orientation & baseline  ✅ (session 1, 2026-11-24)
- [x] Build from source (pip and standalone CMake), record environment gotchas → `BASELINE.md`
- [x] Run the Python suite (799 pass / 34 skip) and the C++ gtests (6/6) → `BASELINE.md`
- [x] Architecture map of every moving part → `ARCHITECTURE.md`
- [x] File-level inventory with review checklist → `INVENTORY.md`
- [x] First probes: NumPy parity grid, NumPy-fallback detector, out-of-core
      consistency, doc-snippet runner → `review/probes/`
- [x] Lint and warnings baselines; ASan+UBSan build
- [x] Findings F-001 … F-024 → `FINDINGS.md`
- [x] Restructure proposal with open decisions → `RESTRUCTURE.md`
- [x] Technical decisions recorded (RESTRUCTURE §3.1)
- [ ] **Maintainer answers the product questions** (RESTRUCTURE §3.2: R1 dtype and
      structure scope, bool semantics). Blocks only the scope pruning in Phase 4.
- [x] Dead-code tooling evaluated plus a coverage-based pipeline → `DEADCODE.md`
- [x] Test-suite redesign strategy plus redundancy measurement → `TESTS.md`
- [x] Full ASan+UBSan pass: no memory errors; 3 UB sites (F-006, F-025)

## Phase 1: Safety net = the NEW test suite core (see TESTS.md)
Rationale: the suite passes while F-001 (int scalar multiply is a no-op) exists, so
"green" doesn't mean "correct". Refactoring without a better net is unsafe.

- [~] **1.1 Test harness:** single `tests/conftest.py`; remove the 15 `sys.path`
      hacks (done: no sys.path edits left); run against an editable install (`pip install -e . --no-build-isolation`);
      stop copying `.so` files into `python/pycauset/` in CI (F-023).
- [x] **1.2 C++ tests in CI:** all 9 gtests wired; `cpp-tests` job in `ci.yml` (session 2;
      first GitHub run still to be observed).
- [x] **1.3 Sanitizers on PRs:** `asan.yml` now runs ASan+UBSan (fatal) on pushes and PRs,
      over both suites. *TSan job still to do.*
- [x] **1.4 Remove stale skips** (F-009); every remaining skip must carry a reason
      plus a FINDINGS id. (Session 9: the old suite and its skips are gone; the new suite's
      only skips are the lazy-off markers and unsigned negation, each with its reason; the
      "NumPy rejects it" skip became an assertion that PyCauset raises too.)
- [~] **1.5 New-suite core** (TESTS.md §3.2), in progress: regressions, scalar ops and the
      dtype-pair conformance grid are done (session 2). Original item: `tests/conftest.py`, `tests/spec.py`,
      `engine/test_conformance.py`, `engine/test_invariants.py`, `engine/test_regressions.py`.
      Details of the grid: promote `review/probes/dtype_matrix_probe.py`
      into a parametrized pytest grid (dtype × op × structure × storage
      {RAM, disk}), asserting values **and** dtypes against NumPy. Initially
      `xfail(strict=True)` for known-open findings, so fixes flip them green.
- [x] **1.6 View-metadata invariants test** (`tests/engine/test_invariants.py`; session 9 added
      windows, conj / H views, Hermitian, scaled and transposed antisymmetric): for every matrix class, element access ==
      NumPy export == `sum` == matmul-with-identity, after each of {scale,
      transpose, conj, slice} (catches the F-001 class of bugs).
- [x] **1.7 Doc-snippet test in CI**: `tests/engine/test_doc_snippets.py` runs every page's
      Python (README, guides, API pages) in a temporary directory (session 9; the probe script
      it replaced is deleted).
- [~] **1.8 Warnings as errors** for our own code (`-Wall -Wextra -Werror` on
      `src/`, not `_deps`), after fixing F-020. Ruff in CI with the current baseline
      frozen (no new findings). Session 9: ruff pyflakes (F) is clean and gated in CI (`lint`
      job); E/I/UP have a backlog (193 long lines, 46 import orders) and are not gated. C++:
      the core library builds clean under -Wall -Wextra (GCC 13); option `PYCAUSET_WERROR`
      (-Wall -Wextra -Wno-unused-parameter -Werror on pycauset_core) is on in the Linux CI
      job. Bindings not yet under it.
- [x] **1.9 Coverage measurement:** Python (`pytest-cov`) and C++ (`gcovr`) once,
      to find untested files and functions (session 9: both runs, results and triage in
      DEADCODE.md; Python 84% lines).

## Phase 1b: Dead code, Tier A (cheap, early; shrinks context for everything after)
- [x] Remove DEADCODE.md §4 Tier A items, batch 1 (session 1: 43 files, −832 lines)
- [~] Add Skylos and ruff-F to CI so new dead code is flagged on PRs (ruff-F done, session 9)

## Phase 2: Critical correctness fixes
Each fix: regression test first (from Phase 1), then fix, then FINDINGS status.
- [x] F-024 eig/eig_skew results share `./_vals` (second call corrupts the first)
- [x] F-001 lazy scalar ignored for integer matrices (ScalarOps)
- [x] F-004 nested ParallelFor deadlock; F-005 exception-path use-after-free
- [x] F-006 signed-overflow UB in integer kernels (all widths, all ops)
- [x] F-025 null memcpy for empty objects
- [x] F-028 sampled overflow preflight removed (determinism policy)
- [x] F-002 operator coverage (`__neg__`, int `__rmul__`, `A - int`, uint64)
- [x] Skew/Arnoldi eigensolvers: Arnoldi rewritten on Spectra (F-030); both tested against NumPy
- [x] F-003 `cond` NumPy default norm, plus other "silently differs from NumPy" defaults
      (do a pass over every function's defaults against NumPy's signature)
- [ ] Anything Phase 1 grids surface (log as new F-ids)

## Phase 3: Deep line-by-line review, area by area
The "literal look at every piece of code". Each area gets a notes file
`review/areas/<area>.md` (template below) and its INVENTORY rows ticked. Order is by
risk: storage and object model first, since everything rests on them.

| # | area | files (see INVENTORY) | approx lines |
|---|---|---|---|
| 3.1 | Storage core: PersistentObject, MemoryMapper, MemoryGovernor, IOAccelerator, StorageUtils, SystemUtils **(reviewed; fixed F-120..F-125; C-12 kept by decision)** | `src/core`, `include/pycauset/core` | 2,300 |
| 3.2 | Object model: MatrixBase/VectorBase and every concrete type; view metadata; CoW **(reviewed; fixed F-117..F-119, F-126; C-10, C-14 open)** | `include/pycauset/{matrix,vector}`, `src/{matrix,vector}` | 3,600 |
| 3.3 | Factory & promotion: ObjectFactory, Promotion/Structure/MatrixType resolvers | `src/core` | 1,100 |
| 3.4 | Dispatch & CPU kernels: ComputeContext, AutoSolver, CpuDevice, **CpuSolver**, ParallelUtils, SIMD | `src/compute` | 7,800 |
| 3.5 | Streaming / out-of-core: StreamingManager (C++ and Py), ComputeWorker, io_observability. **Priority: C++ StreamingManager is 0% executed; is out-of-core really mmap-only?** | | 1,000 |
| 3.6 | Math entry & expressions: LinearAlgebra.cpp, Eigen.cpp, expression templates, LazyMatrix | `src/math`, `matrix/expression` | 1,900 |
| 3.7 | Bindings | `src/bindings*` | 7,300 |
| 3.8 | Python facade & glue: `__init__`, patching, properties, ops, factories, dtypes, lazy_allocation | `python/pycauset` | 6,500 |
| 3.9 | Persistence & caches: persistence, big_blob_cache, linalg_cache, export_guard, container format | | 2,300 |
| 3.10 | Block matrices: blockmatrix, submatrix_view, thunks, dense_python | | 1,200 |
| 3.11 | Physics: causet, spacetime, field, synthetic, Sprinkler.cpp, Spacetime.hpp **(post-R1; smoke tests only before)** | | 2,300 |
| 3.12 | Visualization: vis.py **(R1: lightweight plots only)** | | 435 |
| 3.13 | Build, packaging, CI, release: CMakeLists, pyproject, workflows, tools/*.ps1 | | 1,000 |
| 3.14 | Old tests: superseded by TESTS.md migration (no line-by-line review of old tests) | `tests/` | 16,700 |
| 3.15 | Benchmarks, demos, scripts, tools | | 2,200 |
| 3.16 | CUDA: **inventory only**, no review while on hold | `experimental/cuda` (moved, session 4) | 4,300 |

**Per-file review checklist** (copy into the area notes):
1. Purpose: can you say in one sentence what this file is for? Is it in the right place?
2. Liveness: check `review/deadcode/*_never_executed.txt`. Every never-executed
   function gets a verdict: keep + test, or delete (DEADCODE.md §6)
3. Correctness: view metadata (scalar/transposed/conjugated/offset) honoured;
   all dtypes handled or explicitly rejected; bounds and overflow; NaN/Inf; empty
   (0×n) and 1×1 shapes; non-square.
4. Out-of-core: does it work when operands are disk-backed? Does it materialize
   everything into RAM?
5. Errors: no silent `except: pass` or `catch (...) {}`; error messages say what to do.
6. Concurrency: shared state, singletons, thread pools, GIL release.
7. Performance: obvious waste (copies, per-element virtual calls, Python loops over N²).
8. Duplication: same logic elsewhere? (record for Phase 4)
9. Docs: is it in ARCHITECTURE.md? Do the internals/user docs describe it correctly?
10. Tests: which tests cover it? Is there a gap?

## Phase C: Operation atlas (maintainer request, session 7; design in `features/ATLAS.md`) ⟵ NEXT
Every operation × dtype × structure × view × storage, checked against NumPy and classified as
fast / written-out / generic / numpy-fallback / unsupported, with timings. Generated report
(`review/atlas/ATLAS.md`) + CI tests. Stops the "found one more slow or wrong combination"
cycle by finding all of them at once. Runs before the remaining Phase 4 items, which then
work from the report.
- [x] C.1 Instrumentation (session 8): counted element accessors (`access_counter`, NVI
      wrappers over the `element_*` virtuals), per-operation kernel log
      (`debug_trace::start_log`), NumPy export counter (`export_guard.export_count`).
      Kernel traces are completed lazily: the report lists ops with no trace.
- [x] C.2 Enumerator, oracle, classifier: `tests/atlas/atlas.py` + `report.py` (58,935 cells,
      26 s, deterministic). Runs found F-064..F-079, all fixed: **0 wrong cells** (session 8).
- [~] C.3 Timing pass: `tests/atlas/timing.py` → `review/atlas/TIMING.md` (fast cells, RAM,
      views plain/T/scaled_float/conj; n = 1024 for O(n²), 384 for O(n³), vectors 1M; lazy
      off while timing; per-call overhead table). First run found F-081 (stale caches) and
      F-083 (fast ≠ NumPy speed). Open: wire the ≥ 0.9× NumPy target into the report as a
      gate per operation once the N.2 converter lands.
- [x] C.4 CI: `tests/atlas/test_atlas.py` (fails on any ✗, harness error, or regression vs the
      committed `review/atlas/atlas.json.gz`; KNOWN_GAPS empty). Still to do: delete
      `support_matrix.py`, `tools/check_support_matrix.py`, superseded probes
- [ ] C.5 Work the report down (no ✗ cells; no engine NumPy fallbacks; `generic` only where
      documented)

## Phase N: Normalise at the boundary (session 8; maintainer-approved; a core goal of the review)
Maintainer, session 8: much of the code was AI-generated over about a year, feature on top of
feature without the features being integrated with each other; integrating them ("compounding")
is one of the reasons for this review. Phase N is that work for the operation layer; the same
lens applies to every area (one mechanism instead of N parallel special cases).
Root cause of F-043..F-075 (see LOG session 8): kernels are chosen by one property (dtype or
class) and silently assume others (dense storage, no transpose/conjugation/factor/window,
upper orientation); view flags are optional for every reader; rules are re-implemented in
several places; Python wraps engine calls in silent fallbacks. Cure, mechanism not cells:
- [ ] N.1 Every kernel declares what it accepts (dtype set, structure, "plain" or which flags
      it honours). One entry function per operation normalises any other operand
      (`write_out` / `to_dense` / cast) before dispatch; kernels assert their contract.
      Delete the ad-hoc per-branch guards and the branches the normaliser makes redundant.
      Done: matmul (F-080: 5 rules + 4 kernel contracts replace ~25 front branches and 9
      device fallbacks; atlas matmul 847/847 fast). Next by atlas generic count: norm (674),
      element-wise div/mul/add/sub, scalar ops (~2,600 together), matvec/vecmat (957), neg,
      export/save_load, sum, invert, matrix_power.
- [~] N.2 One definition of view semantics (value = conj(raw · factor), windows, transpose)
      in one header; every reader goes through it or through normalisation. Done (F-084):
      `math/ViewRead.hpp` + typed readers per class behind every conversion. Open: windows of
      structured classes (their `get` ignores windows), the bit dense `get` window bug class,
      and moving the remaining private readers (N.1b list) onto it. Original plan:
      one typed converter `to_dense(any operand, dtype)` with a typed reader per structure
      (dense, triangular, symmetric/antisymmetric, diagonal, identity, float16/complex16
      storage) and the view applied once, no per-element virtual call. Every normaliser uses
      it, so a new structure or view is taught in one place and every op is correct with it
      (fast once a kernel reads it natively). Measured need: F-083 (b); matmul's 290 generic
      cells are all conversions.
- [ ] N.1b Generalise the matmul normaliser into a shared, declarative one (operation =
      contract + plain-dense kernel + optional structure rules), then the ops that still have
      private element readers (atlas generic after N.2): norm 674 (done, F-085: 11 left), div 564, vecmat 511,
      det/slogdet 457 each, matvec 446, mul/sub/add ~250 each, export 187, save_load 172,
      sum 145, add/rsub scalar 129 each. Rewrite `protocols/Adding Operations.md` around it.
- [~] N.6 Per-call overhead (F-083 a; F-096: `a + b` 29 -> 1.9 µs, `a @ b` 49 -> 2.8 µs;
      remaining in `pc.matmul`'s Python layer and the operator wrappers). Original item:: 30-50 µs per `+` / `@` vs NumPy's 0.3-0.7 µs; profile
      the Python wrapper stack (monkeypatches, property patches, io traces) and cut it.
- [x] N.7 Row readers for every class (the speed half of N.2). Done (F-094, F-095): typed
      `read_rows`/`read_diagonal` for every class; det/sum/trace/export/element-wise/bool
      write-out on them; matvec/vecmat through matmul. Session 9 (F-115): bulk stored-value
      reads (`math/RawRows.hpp`) for dense (windows, transposes), triangular, symmetric
      family, diagonal, bit, complex_float16 and closure storage; the atlas has no generic
      per-element cell left (1,556 at the start of N.7). Structured windows read per element
      by design (no class has a window layout).
      Original item:: extend `StridedReader`
      (row-at-a-time, typed, `view_value`) from dense storage to every class through
      `visit_matrix` (packed triangular/symmetric rows are contiguous). Element-wise ops
      (dense and structured results) read rows only; delete `exact_int_reader` and the
      virtual fallback in `binary_op_impl`. Timing pass: typed per-element `get` paths run at
      0.3-0.6x NumPy; contiguous rows are what NumPy-speed needs. Also fold ViewRead.cpp's
      dynamic_cast chain into `visit_matrix` (one dispatch mechanism).
- [x] C.3b Timing harness: integer operands overflow at timing size (values 1-4, n = 384) and
      the int8 linalg recipe adds n to the diagonal; use small values at timing size. (Session
      9: integer linalg operands are sparse 0/1 + 3I; `outer` times n-element vectors, an
      n^2-element pair asked NumPy for terabytes. The 390 error cells were these.)
- [ ] N.5 Structure × dtype completeness (maintainer, session 8: "the heart of PyCauset:
      specialised matrix types for every occasion"; complex symmetric and Hermitian are
      different and both supported as separate flags). Steps, each gated by the atlas
      (new combinations turn from "unavailable" into live cells automatically):
  - [x] N.5.1 One catalogue: a single constexpr predicate of which (structure, dtype) exist
        and a single class map; ObjectFactory, visit_matrix, open_storage (load) and the
        constructor bindings all consult it (today three hand-kept tables).
  - [x] N.5.2 Diagonal and identity for float16, float32, complex64, complex128.
  - [x] N.5.3 Triangular, symmetric, antisymmetric for complex64/complex128.
  - [x] N.5.4 Hermitian and skew-Hermitian (A = Aᴴ, A = -Aᴴ) as flags on the symmetric class
        (as antisymmetric is): packed upper triangle, lower read as conj(±upper), real
        (resp. imaginary) diagonal enforced on set. Complex dtypes only (for real dtypes they
        are symmetric/antisymmetric). Atlas structures `hermitian`, `skew_hermitian`.
  - [x] N.5.5 complex_float16 structured classes: an interleaved `complex_half` element
        type (pair of float16) for TriangularMatrix/Symmetric/Diagonal/Identity<T>.
  - [x] N.5.6 Bit triangular with diagonal (packed rows of n - i bits).
  - [x] N.5.7 UnitVector for every dtype.
  - [x] N.5.8 `Adding Types.md` rewritten around the catalogue (what a new structure or
        dtype must provide: class template, catalogue entry, typed get/set, view reader,
        persistence layout, atlas entry).
- [x] N.3 No silent fallbacks (F-090, F-091): every linear-algebra and vector-product entry
      runs in the engine for every dtype, structure and view; `_internal/ops.py` makes no
      NumPy computation (it imports array-likes only). Pattern: one engine entry that
      normalises to dense in the LAPACK dtype (`to_dense` + `solve_dtype`) and one templated
      LAPACK kernel for float32/float64/complex64/complex128 (`compute/cpu/CpuLinalg.cpp`).
  - [x] N.3b The "properties" layer removed entirely (maintainer decision, session 8: no
        flags, no user notes; F-092). Structure is the class; computed metadata is phase D.
- [ ] N.4 The atlas CI gate proves it: the ✗ count must reach 0 and stay there.

## Phase 4: Structural cleanup (after the RESTRUCTURE decisions)
Behaviour-preserving moves in their own commits, then logic consolidation.
- [x] 4.0 CUDA moved to `experimental/cuda` (code, bindings, `pycauset.cuda`, GPU tests, docs,
      `[gpu]` extra). Dormant core hooks listed in `experimental/cuda/README.md`.
- [ ] 4.1 Dead code Tiers B–D (DEADCODE.md) and debug leftovers (F-019)
  - session 9: tool re-run (vulture, cppcheck). Removed 4 Python + 20 C++ dead helpers and the
    uncalled device methods inverse / multiply_scalar / preferred_precision with their
    CPU, AutoSolver and CUDA implementations (~900 lines). Still open: the C++ coverage run
    (runtime never-executed list), `_safe_rows_cols` copied in four Python files.
- [x] 4.1b Remove `precision_mode` / `set_precision_mode` / `get_precision_mode` end to end
      (code, C++ mode, docs, tests); update the under-promotion warning text
- [x] 4.1d (now tracked by the atlas, Phase C) Dtype matrix completeness: every one of the 16 dtypes works with every other
      in every op of the spec table (the conformance grid is the gate). Session 5: structured
      results (triangular/symmetric/antisymmetric/diagonal) for every real dtype the factory
      instantiates (was int32/float64 only). Gaps left: diagonal/identity float16/float32,
      complex structured storage. Session 9: closed (N.5); the atlas's 103 "unsupported"
      cells are all deliberate raises (integer matmul overflow raises where NumPy wraps;
      signed x uint64 raises, F-035b).
- [ ] 4.1e Fold `ComplexFloat16Matrix/Vector` into the generic templates if feasible
- [x] 4.1c bit/bool arithmetic = integer counting everywhere (RESTRUCTURE §3.2 B, F-013).
      Checked session 9: + - @ and scalar products count in int32, * is AND (the 0/1
      product), unary minus raises as in NumPy; documented in DType System section 6.
- [~] 4.2 Split CpuSolver.cpp, bind_matrix.cpp and `__init__.py` (pure moves). Session 9:
      bind_matrix.cpp 1769 -> 1043 lines (indexing into `src/bindings/matrix_indexing.hpp`);
      CpuSolver.cpp is 864 lines after the earlier kernel-file splits; `__init__.py` waits for
      4.4 (its wrappers need `_OPS_DEPS`, a module split there is a circular import).
  - session 9: CpuSolver.cpp (2,574 lines) -> CpuMatmul.cpp, CpuElementwise.cpp,
    CpuIterative.cpp, CpuSolver.cpp (vector ops, norms, svd, solve) + internal
    CpuKernelsCommon.hpp. bind_matrix.cpp 2,268 -> 1,784 lines: the NumPy importer shared
    with bind_vector.cpp (numpy_import.hpp; bind_vector's copy had diverged). Open: the
    indexing section of bind_matrix.cpp; `__init__.py`, which is not a pure move (its
    functions read module globals such as `debug_mode`): do it with 4.3/4.4.
- [~] 4.3 Single source of truth for operators and properties (RESTRUCTURE §3.1 items 1–2); delete the monkeypatch stack
  - session 7: matrix `+ - * /` (and reflected) are one binding path (`matrix_arith`); vectors
    reuse the matrix machinery (F-058). Open: properties, `__iadd__`, the Python patch stack.
  - session 9, design for the patch stack (`_internal/patching.py` + `track_matrix` /
    `mark_temporary_if_auto` threaded through ops.py, matrix_api.py, `__init__.py`):
    * It wraps 9 fixed methods per class to "register" results; `@`, `/`, `+=`, reflected ops
      and every `pc.*` function not in ops.py are not wrapped, so tracking is partial anyway.
    * Temporary marking is redundant: the engine marks every file it creates (`*.tmp`, from
      `make_unique_storage_file`) temporary itself (`PersistentObject` ctor). The Python rule
      ("under the storage root -> temporary") would also delete a user's own non-.tmp file
      saved there.
    * Tracking feeds two things: the `set_backing_dir` "live matrices" warning (which fires for
      `pc.TriangularMatrix(...)` but not for `pc.matrix(...)` results) and closing
      live matrices before the exit-time `.tmp` sweep (needed on Windows, where a mapped file
      can't be deleted).
    * Plan: a native live-object count (a member counter in PersistentObject: ctor/copy++,
      dtor--) for the warning; at exit the sweep deletes what it can and the engine's
      destructors delete the rest; then delete patching.py, `track_matrix`,
      `mark_temporary_if_auto` and the `__init__` float->int argument coercion (NumPy raises
      for float sizes). Needs a Windows run before merging (exit-time cleanup).
  - session 9, done: patching.py, `track_matrix`, `mark_temporary_if_auto`, the live-object
    WeakSet, the Python temp-file tracker and the float -> int size coercion are removed. The
    `set_backing_dir` warning checks for this process's own `.tmp` files in the old root
    (pid in the name, F-121), and the exit sweep deletes this process's files (temp files
    already go with their storage, F-123). Open: properties, `__iadd__`. Windows: a file still
    mapped at exit cannot be deleted; the next import's sweep removes it (dead pid).
- [ ] 4.4 Python `_internal/` subpackages; `.pyi` generation (F-007)
- [x] 4.5 Retire the old test suite per TESTS.md §4 migration rules (session 9: 72 files retired by coverage, 4 removed with OpRegistry, 19 moved to tests/physics, 9 to tests/blockmatrix; stress substance restored in test_stress.py)
- [x] 4.6a **Structure classes** (§3.1 item 4, revised): `MatrixType`→`Structure` enum; one
      Python class per structure with `.dtype` (F-032 fixed by construction); persistence keyed
      by (structure, dtype); old per-dtype names removed from code, tests and docs
      (migration script: `review/scripts/migrate_class_names.py`).
- [x] 4.6b Naming sweep: milestone names out of code and docs (session 9; the plan files
      under documentation/*/plans keep theirs, they are the milestones).
- [x] 4.7 One StreamingManager (F-100: the Python planner is gone); one Minkowski implementation (native; F-105)
- [x] 4.8 Silent-exception sweep (126 sites). Python first pass done (F-101); C++ sites done
      (session 9: the warning-category and NumPy-scalar fallbacks removed; the remaining
      catch sites rethrow; the parked CUDA hooks in AutoSolver are listed in experimental/cuda).
- [x] 4.9 Per-directory README.md files (what lives here, what doesn't): src/, tests/,
      benchmarks/, tools/ (demos/, experimental/*, review/ had one)

## Phase L: Feature: Lazy evaluation (maintainer request; spec in `features/LAZY.md`)
Views and deferred expressions for every dtype, structure and vector. Rule: laziness is
unobservable except through speed and memory. Order: L0 is Phase 4.3 + 4.6 + F-043;
then L1 → L2 → F1 → L3 → L4 → L5 (only if measured worthwhile) → L6.
- [ ] L0 Prerequisites: structure classes (4.6a ✅), single operator path
      (4.3), views read at kernel speed (F-043)
- [ ] L1 `ViewSpec`: typed exact factors, scale/negate/conj/transpose/cast views for all
      16 dtypes × all structures × matrices and vectors; free transposes (F-044); streaming
      reads views (F-040 follow-up); lazy≡eager oracle switch `_set_lazy(False)`
- [ ] L2 Closures `A ± Aᵀ` over triangular storage, diagonal shift `A + cI`; eigensolvers
      consume them without materializing (Pauli–Jordan)
- [ ] L3 `ExprNode`: sums, products with chain ordering, powers, inverse-as-solve;
      matvec/trace/sum/diag consumers; asymptotic tests via kernel trace
- [ ] L4 Property propagation through views and nodes (needs C++-owned properties)
- [ ] L5 Element-wise fusion via sum nodes (only after F-012; benchmark gate)
- [ ] L6 Constant matrices, slice views

## Phase F1: Feature: Value bounds (maintainer request; spec in `features/VALUE_BOUNDS.md`)
Runs after Phase 4.3 (single C++ operator/result-type path) and after lazy L1/L2 (it
shares the `ViewSpec`), before Phase 6.
- [ ] F1.1 `ValueBounds` on PersistentObject + persistence + `A.value_bounds`
- [ ] F1.2 Sources (bit=[0,1], factories, NumPy import min/max, setitem widening, unknown on opaque writes)
- [ ] F1.3 Propagation rules for every op in the spec table
- [ ] F1.4 Bounds-aware integer result dtype in PromotionResolver; delete the heuristic overflow preflight
- [ ] F1.5 Soundness invariant in the conformance grid; memory benchmark on causal `C @ C`
- [ ] F1.6 Docs page + NumPy-divergence note

## Phase D: Feature: Derived-value cache (maintainer request, session 8; spec in `features/DERIVED_CACHE.md`)
Computed results (det, trace, eigenvalues, norms, rank, ...) are kept with the matrix and saved
with it; any change to the storage discards them first; scaled / conjugated / transposed views
derive their answer from the cached one (det(4A) = 4^n det(A)). After N.7 and with F1 (both
are engine-owned metadata on the storage).
- [x] D.1 Storage version counter; every mutable path invalidates (test walks every writer).
      Session 9: `MemoryMapper::version`, bumped by `PersistentObject::ensure_unique` (the
      copy-on-write gate every writer passes); `tests/engine/test_derived_cache.py`.
- [~] D.2 Cache map + view rules for scalar and vector quantities. Session 9: the map on the
      storage and `cached()` (math/DerivedCache.hpp) for det, slogdet, trace, sum, matrix
      norms, rank. Keyed on the exact view instead of the spec's view rules (bit-identical to
      recomputation; the rules round differently). eigvals / eigvalsh too (a copy-on-write clone
      per call); svdvals open.
- [x] D.3 Persistence with invalidate-before-mutate on memory-mapped files (session 9: a
      `derived` metadata list written by save, restored by load; loaded files are private
      copy-on-write mappings, so a write copies and drops the entries and the file never
      changes; the crash case of the spec does not arise)
- [ ] D.4 Matrix-valued results (inverse, eigenvectors, factors) as container sections
- [ ] D.5 Closures and expression nodes (with phase L)

## Phase 5: Documentation truth pass
- session 9 (started with the pages physics touched): README, Quickstart, User Guide,
  Examples, Saving and Loading, the three index pages, save/load, Public API Contract. Their
  code runs in `tests/engine/test_doc_snippets.py` (add each page as it is rewritten).
  Then internals/ (Compute Architecture, Algorithms, Memory and Data, Block Matrices) and
  dev/ (all pages; four historical ones deleted), BENCHMARKS.md, the Visualization guide.
  Every public name now needs a doc page (test). Next: documentation/project (Philosophy,
  Contributing, protocols, Public API Contract) and the R2 plans under project/plans.
- [x] 5.1 Fix every broken snippet (F-008); the snippet test is green (every page with code,
      81 pages, session 9). Includes removing GPU
      prose from guides/internals (pages moved in 4.0; prose left for this pass)
- [ ] 5.2 API reference generated from docstrings, one page per public function/class
      (NumPy-style); no pages for missing symbols. **Style rules:** lead with what it
      does in one sentence; signature, parameters, returns, raises; one minimal
      runnable example; NumPy differences called out explicitly. No marketing words,
      no restating the name as a sentence, no "powerful/seamless/robust", no padding
      sections. Every example is executed in CI.
- [x] 5.3 Delete historical plans/archive (done); one ROADMAP (done, session 9:
      `documentation/project/Roadmap.md` replaces internals/plans/TODO.md, whose "R1 is
      complete" was no longer true; the R2 physics plans moved to
      `experimental/physics/docs/plans/` with the other parked physics plans); merge dev/ +
      internals/ (done: one `documentation/internals/` with one index, "Developers" in the nav)
- [~] 5.4 README and BENCHMARKS claims re-measured (BENCHMARKS done on Linux, F-108; the
      maintainer's Windows numbers next) ("≥0.90× NumPy", "bigger than RAM")
- [x] 5.5 ARCHITECTURE.md moves into `documentation/internals/` as the canonical map (rewritten
      to the code, session 9)

## Phase 6: Performance (only after correctness)
- [ ] 6.0 One OpenBLAS per process: link against NumPy's bundled scipy-openblas (or share its
      thread pool) instead of the system OpenBLAS; two spinning pools cost up to 50x on BLAS
      calls when NumPy is imported first (F-096). Today: a warning and an import-order rule.
- [x] 6.0a LAPACK without row-major LAPACKE transposes (F-108; done: see BENCHMARKS.md): solve / invert / det / eigh at
      0.5-0.7x NumPy, the column-major call on the same buffer is 2x faster than NumPy
- [x] 6.0b `lstsq` through `?gelsd` (NumPy's routine) instead of `pinv(a) @ b` (0.65x)
- [~] 6.1 Elementwise/vector parity (F-012): profile the wrapper stack vs the kernels.
      Session 9: char-type aliasing (ParallelForRange), packed structured kernels (F-099),
      bulk stored-value reads (F-115), compile-time operator loops, structured matvec over
      packed storage, chunked sums, unit-vector dot, gemv forwarding. Left: per-call
      overhead (~2 us of Python layers per op), complex32 and bool structured operands,
      symmetric matvec (0.3x: needs a blocked symv-style kernel), page-faulting fresh results.
- [ ] 6.2 Thread-pool consolidation (F-014)
- [x] 6.3 Out-of-core paths for eigh/svd/rank (F-011) or document them as in-RAM only
      (documented, session 9)
- [ ] 6.4 Large-N out-of-core benchmark on a real disk (README's headline claim)

---

## Potential future features (not scheduled; maintainer decides when)
- Per-block payload checksum in the container (detects modification of a saved file by
  another program; lets the derived-value cache survive that case). DERIVED_CACHE.md, limits.

## Area-notes template (`review/areas/<area>.md`)
```
# Area: <name>
Reviewer(s): <agent/session>   Status: in progress | done
Files: <list, with lines>
## What it does (2–5 sentences, then update ARCHITECTURE.md if it differs)
## File-by-file notes
### path/to/file  (checklist items 1–10, terse)
## Findings raised: F-0xx, ...
## Proposed changes (feed into Phase 4)
```
