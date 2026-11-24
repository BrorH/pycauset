# Dead-Code Program

**Why (maintainer, 2026-11-24):** in an agent-written codebase, dead code costs
twice: it burns context window and it misleads the next agent into extending or
"fixing" code that doesn't matter. Removing it is a first-class goal of this
overhaul, not an afterthought.

**Definition used here.** Code is dead if removing it changes nothing a user can
observe. That covers four kinds:
1. **Unreferenced**: nothing calls it (e.g. `WorkloadPartitioner`, `lazy_ufunc.py`).
2. **Referenced but never executed**: called only from branches that never fire
   (e.g. the pure-Python fallback matrix in `matrix_api.Matrix`, reached from
   `pc.matrix` branches that never trigger).
3. **Superseded**: a newer path does the same job and the old one survives as a
   fallback, shim or alias (e.g. two StreamingManagers, Minkowski in Python *and*
   C++, NumPy fallbacks next to native kernels).
4. **Unwanted surface**: public features nobody needs for the release. Only the
   maintainer can call these, so tools surface them as questions (§5).

---

## 1. Tools evaluated (2026-11-24)

| Tool | Language | Kind | Verdict on this repo |
|---|---|---|---|
| **coverage.py** (`--cov-context=test`) | Python | runtime: never executed | **Best signal.** Finds kind 2. Per-test contexts also measure test redundancy |
| **gcov + gcovr** (`--coverage` build) | C++ | runtime: never executed | **Best signal for C++.** Template copies merged per source function by `review/scripts/cpp_unexecuted.py` |
| **Skylos** 4.x | Python | static | Good: 13 findings, almost all real, low noise. Recommended static gate |
| **Vulture** 2.16 | Python | static | Noisy: ~60 findings at 60% confidence, mostly public API false positives. Skylos supersedes it |
| **deadcode** | Python | static | Installed, not yet compared. Optional |
| **cppcheck** `unusedFunction` | C++ | static | Medium: 47 findings, several false positives (templates, function pointers). Useful **intersected** with coverage |
| ruff `F401/F841` | Python | static | Unused imports and variables; cheap, run in CI |
| `-Wunused-*` (GCC) | C++ | static | Unused variables (F-020) |

Industry practice (Microsoft, GitHub's gh-aw dead-code agent, Uber's AutoClean, the
DCE-LLM paper) converges on the same pattern: deterministic detectors propose, an
agent removes, and **build + tests verify every removal**, with a human deciding the
judgment calls. No off-the-shelf "LLM dead-code product" beats that loop.

## 2. Pipeline (re-run after each cleanup batch)
```bash
# Python, static
skylos python/pycauset --json > skylos.json
# Python, runtime (also gives test redundancy)
python -m pytest tests/python --cov=python/pycauset --cov-context=test --cov-report=
python review/scripts/coverage_analysis.py .coverage functions   # never-executed functions
python review/scripts/coverage_analysis.py .coverage tests       # unique lines per test file
# C++, static
cppcheck --enable=unusedFunction --std=c++20 --language=c++ -Iinclude --quiet \
  src/core src/matrix src/vector src/compute src/math src/causet src/bindings src/bindings.cpp include
# C++, runtime: coverage build (Debug, --coverage, ~15 min, needs ~9 GB RAM for the big TUs),
# then run the Python suite AND demos AND gtests against it (see BASELINE.md), then:
gcovr -r . build/cov --filter src/ --filter include/ --exclude src/accelerators/ --json cppcov.json
python review/scripts/cpp_unexecuted.py cppcov.json
```
"Usage" means: the test suite, the demos, the doc snippets and the gtests. A
function none of those execute is a candidate. Before removing it, also check that
it isn't a public API someone would reasonably call (§5).

Snapshots of every report are in `review/deadcode/` (taken 2026-11-24 at `046c5ab`,
Python suite + demos, **without** gtests).

## 3. Headline numbers (2026-11-24)
- **Python:** 76% line coverage. **87 functions (~464 lines) never execute.**
  Two modules are 0% (`lazy_ufunc.py`, `interop.py`). Skylos: 13 static findings.
- **C++:** 35% line / 45% function coverage (template-inflated). **489 of 1,562
  source functions never execute.** 24 are also statically unreferenced (cppcheck ∩
  coverage).
- **`src/compute/StreamingManager.cpp`: 0% executed.** The C++ out-of-core tiling
  engine never runs in tests or demos. Either out-of-core works purely through mmap
  paging and this engine is dead weight, or the headline feature is untested.
  Resolving this is a Phase 3.5 priority.
- Structured types barely run: IdentityMatrix 3%, DiagonalMatrix 5%,
  SymmetricMatrix 7%, TriangularMatrix (non-bit) 11%, AutoSolver 22%.

## 4. Candidate tiers

### Tier A: remove now (unreferenced AND never executed, or clearly obsolete)
**Batch 1 done (session 1, 2026-11-24)** ✅: everything below except where marked.
Lessons: (1) a candidate flagged by tools can still have a live reader in the same
file (`g_constructing`, kept); (2) code used only by its own unit test is dead
(`load_matrix`, `load_vector`, `flush_async`, `promote_to_ram` removed with their tests);
(3) never use blind text substitution for removals (a duplicated line broke 47 tests
until reverted); use the AST/brace-matching helpers.

Python: `_internal/lazy_ufunc.py`; `_internal/interop.py` (`patch_interop`,
`_array_ufunc`); `ops._streaming_eigh`; unreachable code after `return` in
`ops.eigvals`; `_storage.StorageRegistry`, `tracked_temp_files`;
`streaming_manager.default_manager`; `io_observability.default_instance`;
`support_matrix.summarize`; unused vars from `skylos_python.txt`.
C++ (cppcheck ∩ coverage, 24): `WorkloadPartitioner.{hpp,cpp}`, `src/main.cpp`,
`AutoSolver::select_device{,_for_matrix}`, `CpuSolver::matmul_dense`,
`IOAccelerator::flush_async`, `ComputeContext compute_context_is_constructing`,
`SystemUtils get_available_ram/get_total_ram`, `LinearAlgebra is_triangular`,
`binding_warnings warn_with_category/warn_once`, `ObjectFactory::load_matrix/load_vector`
(check the persistence path first), `PersistentObject::promote_to_ram`.
**Not yet done:** `MemoryGovernor`'s pinning API is GPU infrastructure; it moves with
CUDA to `experimental/` rather than being deleted. `set_temporary_file` (never executed,
but wired into Runtime) is Tier B.

### Tier B: referenced but never executed (decide: test it or delete it)
- `matrix_api.Matrix` pure-Python matrix implementation (~60 lines, reached from
  `pc.matrix`/`pc.zeros` fallback branches at `__init__.py:1319,1376,1532` that
  never fire) and the empty `pc.TriangularMatrix` shim.
- `lazy_allocation` dtype-less `zeros()` wrapper: 18 operator methods never run.
- `pc.divide`: public, never called by any test.
- `CpuSolver`: `gemm` (80 lines), `multiply_scalar` (36), several `matmul`
  type-combination branches, `batch_gemv`. (`eigvals_arnoldi` was rewritten on
  Spectra in session 3 and is now live and tested.)
- `StreamingManager.cpp` + `CpuComputeWorker` (see §3).
- All of `review/deadcode/cpp_never_executed.txt` and `py_never_executed.txt`,
  reviewed file by file during PLAN Phase 3. Each reviewer records keep/test/delete.

### Tier C: superseded paths (consolidate, then delete the loser)
- Two StreamingManagers (C++ vs `_internal/streaming_manager.py`).
- Operator semantics in pybind overloads *and* Python wrapper layers.
- Properties in C++ flags *and* `properties.py`.
- NumPy-fallback branches in `ops.py` next to native kernels (F-011).
- `ComputeWorker` / `ComputeWorkerExtensionTemplate.hpp` "how to add an op"
  scaffolding, if StreamingManager goes.

### Tier D: unwanted surface (maintainer decides; see §5)

## 5. Questions for the creative director (surface decisions)
These are the biggest levers: every dtype or structure we drop deletes template
code across ~40 C++ classes and their bindings.
1. **Dtypes for R1.** Today 16 dtypes: bit/bool, int8/16/32/64, uint8/16/32/64,
   float16/32/64, complex16/32/64 (complex16 is hand-written). Which does causal-set
   work actually need?
2. **Structures for R1.** Dense, triangular, bit, symmetric/antisymmetric,
   diagonal, identity, unit vector, block matrices, lazy expressions.
3. **Rarely used features:** `precision_mode`, memory hints (`AccessPattern`,
   `MemoryHint`), IO traces (`last_io_trace`, `clear_io_traces`, streaming
   thresholds), `compute_k`/`compute_k_matrix`, `eigvals_skew`/`eig_skew`,
   `eigvals_arnoldi`, `convert_file`, the npz/npy helpers, all six `bitwise_*`.

Record answers here with a date. Anything cut is removed completely (code, tests,
docs, plans).

**Answers (maintainer, 2026-11-24):** keep **all 16 dtypes** (under-promotion is the
point; int64 is the ceiling). Keep `eigvals_skew`, `eig_skew` and `eigvals_arnoldi`;
their never-executed C++ code moves from "candidate" to **"must be tested"**. Defer
and remove `precision_mode` (+ set/get). Structures: judged one by one in Phase 3.
Other rarely used features: tech lead proposes per feature in Phase 3.

## 6. Removal procedure (for agents)
1. Pick one tier and one area. Small batches; one commit per logical removal.
2. Grep for the symbol across `src include python tests benchmarks tools demos
   documentation` (including docs and plans).
3. Delete the code **and** its tests, docs pages, `__all__` entries, `.pyi` lines and
   plan references.
4. Build, then run the suite (`pytest -q`) and the gtests. Any failure means it
   wasn't dead: revert, record why in the candidate list.
5. Note the removal in LOG.md; tick it in this file.
6. Re-run the §2 pipeline after each area to refresh the snapshots.
