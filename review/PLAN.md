# Codebase Review & Health Overhaul: Master Plan

**Goal:** every line of PyCauset reviewed, every claim in the docs true, the
repository cleanly organized, and the engine "smooth as butter": no silent wrong
answers, no hangs, NumPy-compatible where it claims to be, fast where it claims to
be. Expected duration: weeks. Many agents, and the maintainer, will work on it.

**Scope:** everything (C++ engine, bindings, Python layer, physics suite, tests,
benchmarks, tools, build and CI, docs). **Out of scope:** GPU/CUDA work (on hold;
moves to `experimental/`, RESTRUCTURE §3.1) and new features.

**Priority for R1 (maintainer, 2026-11-24):** the *engine*, meaning linear
algebra and "NumPy for causal sets". The physics suite is **deprioritized**: smoke
tests only, and fix glaring bugs if found, but no deep review or new spacetimes
until after R1.

**Companion strategies:** `TESTS.md` (new compact test suite) and `DEADCODE.md`
(dead-code program). Both run *through* the phases below.

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

- [ ] **1.1 Test harness:** single `tests/conftest.py`; remove the 15 `sys.path`
      hacks; run against an editable install (`pip install -e . --no-build-isolation`);
      stop copying `.so` files into `python/pycauset/` in CI (F-023).
- [x] **1.2 C++ tests in CI:** all 9 gtests wired; `cpp-tests` job in `ci.yml` (session 2;
      first GitHub run still to be observed).
- [x] **1.3 Sanitizers on PRs:** `asan.yml` now runs ASan+UBSan (fatal) on pushes and PRs,
      over both suites. *TSan job still to do.*
- [ ] **1.4 Remove stale skips** (F-009); every remaining skip must carry a reason
      plus a FINDINGS id.
- [~] **1.5 New-suite core** (TESTS.md §3.2), in progress: regressions, scalar ops and the
      dtype-pair conformance grid are done (session 2). Original item: `tests/conftest.py`, `tests/spec.py`,
      `engine/test_conformance.py`, `engine/test_invariants.py`, `engine/test_regressions.py`.
      Details of the grid: promote `review/probes/dtype_matrix_probe.py`
      into a parametrized pytest grid (dtype × op × structure × storage
      {RAM, disk}), asserting values **and** dtypes against NumPy. Initially
      `xfail(strict=True)` for known-open findings, so fixes flip them green.
- [ ] **1.6 View-metadata invariants test:** for every matrix class, element access ==
      NumPy export == `sum` == matmul-with-identity, after each of {scale,
      transpose, conj, slice} (catches the F-001 class of bugs).
- [ ] **1.7 Doc-snippet test in CI** (`review/probes/run_doc_snippets.py` → `tests/docs/`),
      with a file-size rlimit.
- [ ] **1.8 Warnings as errors** for our own code (`-Wall -Wextra -Werror` on
      `src/`, not `_deps`), after fixing F-020. Ruff in CI with the current baseline
      frozen (no new findings).
- [ ] **1.9 Coverage measurement:** Python (`pytest-cov`) and C++ (`gcovr`) once,
      to find untested files and functions → record in INVENTORY.md.

## Phase 1b: Dead code, Tier A (cheap, early; shrinks context for everything after)
- [x] Remove DEADCODE.md §4 Tier A items, batch 1 (session 1: 43 files, −832 lines)
- [ ] Add Skylos and ruff-F to CI so new dead code is flagged on PRs

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
| 3.1 | Storage core: PersistentObject, MemoryMapper, MemoryGovernor, IOAccelerator, StorageUtils, SystemUtils | `src/core`, `include/pycauset/core` | 2,300 |
| 3.2 | Object model: MatrixBase/VectorBase and every concrete type; view metadata; CoW | `include/pycauset/{matrix,vector}`, `src/{matrix,vector}` | 3,600 |
| 3.3 | Factory & promotion: ObjectFactory, Promotion/Structure/MatrixType resolvers | `src/core` | 1,100 |
| 3.4 | Dispatch & CPU kernels: ComputeContext, AutoSolver, CpuDevice, **CpuSolver**, ParallelUtils, SIMD | `src/compute` | 7,800 |
| 3.5 | Streaming / out-of-core: StreamingManager (C++ and Py), ComputeWorker, io_observability. **Priority: C++ StreamingManager is 0% executed; is out-of-core really mmap-only?** | | 1,000 |
| 3.6 | Math entry & expressions: LinearAlgebra.cpp, Eigen.cpp, expression templates, LazyMatrix | `src/math`, `matrix/expression` | 1,900 |
| 3.7 | Bindings | `src/bindings*` | 7,300 |
| 3.8 | Python facade & glue: `__init__`, patching, properties, ops, factories, dtypes, lazy_allocation | `python/pycauset` | 6,500 |
| 3.9 | Persistence & caches: persistence, big_blob_cache, linalg_cache, export_guard, container format | | 2,300 |
| 3.10 | Block matrices: blockmatrix, submatrix_view, thunks, dense_python | | 1,200 |
| 3.11 | Physics: causet, spacetime, field, synthetic, Sprinkler.cpp, Spacetime.hpp **(post-R1; smoke tests only before)** | | 2,300 |
| 3.12 | Visualization: vis.py **(post-R1)** | | 435 |
| 3.13 | Build, packaging, CI, release: CMakeLists, pyproject, workflows, tools/*.ps1 | | 1,000 |
| 3.14 | Old tests: superseded by TESTS.md migration (no line-by-line review of old tests) | `tests/` | 16,700 |
| 3.15 | Benchmarks, demos, scripts, tools | | 2,200 |
| 3.16 | CUDA: **inventory only**, no review while on hold | `src/accelerators/cuda` | 4,300 |

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

## Phase 4: Structural cleanup (after the RESTRUCTURE decisions)
Behaviour-preserving moves in their own commits, then logic consolidation.
- [ ] 4.1 Dead code Tiers B–D (DEADCODE.md) and debug leftovers (F-019)
- [x] 4.1b Remove `precision_mode` / `set_precision_mode` / `get_precision_mode` end to end
      (code, C++ mode, docs, tests); update the under-promotion warning text
- [ ] 4.1d Dtype matrix completeness: every one of the 16 dtypes works with every other
      in every op of the spec table (the conformance grid is the gate)
- [ ] 4.1e Fold `ComplexFloat16Matrix/Vector` into the generic templates if feasible
- [ ] 4.1c bit/bool arithmetic = integer counting everywhere (RESTRUCTURE §3.2 B, F-013)
- [ ] 4.2 Split CpuSolver.cpp, bind_matrix.cpp and `__init__.py` (pure moves)
- [ ] 4.3 Single source of truth for operators and properties (RESTRUCTURE §3.1 items 1–2); delete the monkeypatch stack
- [ ] 4.4 Python `_internal/` subpackages; `.pyi` generation (F-007)
- [ ] 4.5 Retire the old test suite per TESTS.md §4 migration rules
- [ ] 4.6 Naming sweep: milestone names out; class renames (§3.1 item 4). **Together with
      F-032:** register a Python class for every instantiation the factory can create
      (generated from one dtype×structure table: bindings, persistence class map, facade,
      docs), named in the new scheme (e.g. `TriangularInt16Matrix`). Prerequisite for
      value bounds (F1), which will produce int8/int16 triangular counts.
- [ ] 4.7 One StreamingManager; one Minkowski implementation path
- [ ] 4.8 Silent-exception sweep (126 sites)
- [ ] 4.9 Per-directory README.md files (what lives here, what doesn't)

## Phase L: Feature: Lazy evaluation (maintainer request; spec in `features/LAZY.md`)
Views and deferred expressions for every dtype, structure and vector. Rule: laziness is
unobservable except through speed and memory. Order: L0 is Phase 4.3 + 4.6 + F-043;
then L1 → L2 → F1 → L3 → L4 → L5 (only if measured worthwhile) → L6.
- [ ] L0 Prerequisites: generated dtype×structure table (4.6/F-032), single operator path
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

## Phase 5: Documentation truth pass
- [ ] 5.1 Fix every broken snippet (F-008); the snippet test is green
- [ ] 5.2 API reference generated from docstrings, one page per public function/class
      (NumPy-style); no pages for missing symbols. **Style rules:** lead with what it
      does in one sentence; signature, parameters, returns, raises; one minimal
      runnable example; NumPy differences called out explicitly. No marketing words,
      no restating the name as a sentence, no "powerful/seamless/robust", no padding
      sections. Every example is executed in CI.
- [ ] 5.3 Delete historical plans/archive; one ROADMAP; merge dev/ + internals/
- [ ] 5.4 README and BENCHMARKS claims re-measured ("≥0.90× NumPy", "bigger than RAM")
- [ ] 5.5 ARCHITECTURE.md moves into `documentation/internals/` as the canonical map

## Phase 6: Performance (only after correctness)
- [ ] 6.1 Elementwise/vector parity (F-012): profile the wrapper stack vs the kernels
- [ ] 6.2 Thread-pool consolidation (F-014)
- [ ] 6.3 Out-of-core paths for eigh/svd/rank (F-011) or document them as in-RAM only
- [ ] 6.4 Large-N out-of-core benchmark on a real disk (README's headline claim)

---

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
