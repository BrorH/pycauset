# Findings Log

Every concrete issue found during the review. One entry per issue, with a stable
ID (`F-NNN`, never reused), a severity, evidence or a repro, and a status.

**Severity**
- **S1 critical**: silent wrong results, data loss, crash or hang in normal use.
- **S2 major**: a documented or NumPy-expected feature fails loudly, or UB that
  happens to work today.
- **S3 minor**: inconsistency, poor UX, perf gap, stale artifact.
- **S4 hygiene**: dead code, naming, lint, docs typos.

**Status**: `open` → `in-progress (who)` → `fixed (commit)` / `wontfix (why)`.

When you fix one: add a regression test, set the status to the commit hash, and keep
the entry (the history is useful). New findings go at the bottom with the next free ID.

---

## S1 — critical

### F-001 Integer matrix × scalar silently returns the unscaled matrix
- **Status:** fixed (`fix(F-001,F-002,F-003)`): new `src/math/ScalarOps.cpp`. Scope was wider
  than first recorded: every integer AND bit dtype (incl. the causal matrix) gave wrong
  results for non-trivial scalars. Tests: `tests/engine/test_scalar_ops.py` (566 cases vs
  NumPy, values + dtype), `test_regressions.py::test_F001_*`.
- **Where:** `DenseMatrix<T>::multiply_scalar` (`include/pycauset/matrix/DenseMatrix.hpp:229`)
  makes an O(1) view with `scalar_ *= factor`. For `IntegerMatrix` (int32) the
  readers don't apply `scalar_`: `__getitem__`, NumPy export and matmul ignore it,
  while `pc.sum` honours it.
- **Repro:**
  ```python
  A = pc.matrix(np.array([[1, 2], [3, 4]], dtype=np.int32))
  B = A * 3          # also A*3.0, A*0, A/2 -> all "unchanged"
  B[0, 1]            # 2   (expected 6)
  pc.to_numpy(B)     # [[1,2],[3,4]]
  pc.sum(B)          # 30  (correct: sum applies the scalar)
  B.scalar           # (3+0j)
  ```
- **Why tests missed it:** the suite never compares int32 scalar-multiply results
  element by element. Found by `review/probes/dtype_matrix_probe.py`.
- **Root cause (session 1):** every integer dtype scales via `DenseMatrix<T>::multiply_scalar`,
  which only sets the lazy `scalar_` field (`allow_lazy_scalar` is false for ints, so the
  binding takes the "eager" branch, which isn't eager). Some integer classes' readers
  apply `scalar_` (int8/16/64 export) and others don't (int32 `__getitem__`, export, matmul).
- **Decided design (tech lead, session 1):** a lazy `scalar_` cannot express NumPy's result
  dtypes for integers (int32×3 → int32, int32×3.0 → float64, int32/2 → float64), so
  **integer and bit matrices never carry a lazy scalar**. Scalar ops on them compute the
  result for real, with NumPy result dtypes:
  - int dtype × Python int → same dtype, wrapping multiply (`scalar::WrapMultiplies`, F-006);
  - int dtype × Python float, and int / anything → float64;
  - int × complex → complex128;
  - bit × int → integer counting dtype (F-013), bit × float → float64.
  Implement it as a typed C++ kernel (no round-trip through `double`, which loses
  int64 precision above 2^53) that honours transposed/triangular/bit views, inside the
  single C++ operator table (PLAN 4.3) or just before it. `set_scalar()` on integer
  classes must raise. Float and complex matrices keep the O(1) lazy scalar.
- **Tests waiting:** `tests/engine/test_regressions.py::test_F001_*` (strict xfail).

### F-004 Nested `ParallelFor` deadlocks
- **Status:** fixed (`fix(F-004,F-005)`): nested calls run inline on pool workers;
  gtests `NestedParallelism`, `NestedParallelismManyLevels` (verified to hang on old code)
- **Where:** `include/pycauset/core/ParallelUtils.hpp`. Outer tasks occupy every
  pool worker and block in `fut.get()` on inner tasks queued behind them.
- **Repro:** `tests/test_parallel_utils.cpp::NestedParallelism` hangs forever. That
  test is not wired into CMake, so nobody saw it.
- **Impact:** any kernel that calls a `ParallelFor`-using helper from inside a
  `ParallelFor` body hangs. No such nesting is confirmed in production yet; that
  check is on the plan.
- **Fix direction:** run inline when already on a pool thread (a thread_local flag),
  or let the caller participate in the work. Wire the test into CMake.

### F-024 `eig` / `eig_skew` results all share one file `./_vals`; a second call corrupts the first
- **Status:** fixed (`fix(F-024)`); regression tests `tests/engine/test_regressions.py::test_F024_*`
- **Where:** `src/math/LinearAlgebra.cpp:1074-1115` builds backing names as
  `result_file + "_vals"` / `"_vecs"`. With the default `result_file = ""` that is
  the literal file `_vals` (and `_vecs`) **in the current working directory**,
  not an auto-named temp file. Every eig result is mmapped onto the same file.
- **Repro (default settings):**
  ```python
  w1, _ = pc.eig(pc.matrix(A)); a = pc.to_numpy(w1, allow_huge=True).copy()
  w2, _ = pc.eig(pc.matrix(B))
  np.allclose(a, pc.to_numpy(w1, allow_huge=True))   # False: w1 now holds B's data
  w1.get_backing_file()                               # '_vals'
  ```
- Also: even 6×6 eig results are disk-backed, so `np.asarray(w)` hits the export
  guard (F-016). Two processes in the same directory would corrupt each other too.
- **Patchwork marker:** `.gitignore` lines 128–129 ignore `/_vals` and `/_vecs`.
  The symptom (stray files after the test run) was hidden instead of fixed.
- The neighbouring `lu`/`qr`/`svd` (lines 950–996) already use
  `result_file.empty() ? "" : result_file + "_u"`. The same bug was fixed there and
  missed in the eig family.
- **Fix direction:** pass `""` through so ObjectFactory picks a unique temp name;
  audit every other `result_file + "<suffix>"` concatenation; remove the
  `.gitignore` lines. Regression test: two live eig results stay distinct.

## S2 — major

### F-002 Scalar/operator coverage is inconsistent across dtypes
- **Status:** fixed (`fix(F-001,F-002,F-003)`). Root cause: pybind tries overloads in
  registration order, and catch-all `py::object` overloads registered *before* the typed
  ones returned NotImplemented for plain Python numbers (`A - 1`, `3 * A`). Also: the
  `int64_t` caster accepted NumPy ints via `__index__`, discarding their strong dtype.
  `__neg__`/`__pos__` added. The uint64 error was actually F-033 (export).
- `-A` (`__neg__`) is not defined for **any** matrix type: `TypeError: bad operand type for unary -`.
- `3 * A` (int on the left) raises `TypeError` for int8/16/32/64 and float32/64,
  while `2.0 * A` works.
- `A - 1` raises `TypeError` for int32, yet `1 - A` works.
- `A / 2` on int32 returns the matrix unchanged (part of F-001) instead of true
  division to float like NumPy.
- uint64 `A - B` raises `OverflowError: Python int too large to convert to C long`.
- **Repro:** `review/probes/dtype_matrix_probe.py --verbose`, plus the snippets above.
- **Root cause (likely):** operator semantics are split across about 8 pybind
  overloads per operator *and* two layers of Python wrappers (ARCHITECTURE §4.2).

### F-003 `pc.cond` uses the Frobenius norm; NumPy's default is the 2-norm
- **Status:** fixed (`fix(F-001,F-002,F-003)`): NumPy definition for all `p`; singular → inf.
  An old test asserted the wrong (Frobenius) value and was corrected.
- `pc.cond([[4,1],[2,3]])` = 3.0, `np.linalg.cond` = 2.618. Any `p=` raises
  NotImplementedError. This violates the "speaks NumPy" contract silently.
- **Where:** `python/pycauset/_internal/ops.py:852`.

### F-005 `ParallelFor` exception path can use-after-free
- **Status:** fixed (`fix(F-004,F-005)`): waits for all tasks, then rethrows the first
  error; stops handing out chunks after a failure; `global_num_threads` is atomic.
  gtest `ExceptionWaitsForAllWorkers` (verified to fail on old code)
- When one chunk throws, `fut.get()` rethrows in the caller while other workers keep
  running `func`, which usually captures the caller's stack by reference. The caller
  unwinds and those references dangle. Fix: wait for all futures, then rethrow the
  first exception.
- Also: `ThreadPool::global_num_threads` is a non-atomic static read across threads.

### F-006 Signed-integer-overflow UB in int32 elementwise add
- **Status:** fixed (`fix(F-006,F-025)`): `scalar::WrapPlus/WrapMinus/WrapMultiplies`
  (ScalarUtils.hpp) do integer math in the unsigned type; used by all 42 functor sites and
  the scalar fallbacks in CpuSolver. UBSan run: 0 reports. Tests `test_F006_*`.
- **Where:** `src/compute/cpu/CpuSolver.cpp:165` (`scalar_add<int>`), reached from
  `test_edge_cases_core.py::TestIntegerOverflowPolicy::test_elementwise_int32_add_wraps`.
- The project's policy is NumPy-style wrapping, but it relies on signed overflow,
  which is undefined behaviour. It works today only by compiler accident. The other
  integer kernels (sub, mul, scalar ops, all int widths) most likely share this.
- Full sanitized run (2026-11-24) also hits int32 multiply overflow (`46341*46341`, via
  `std::multiplies` in `stl_function.h:287`), from `test_elementwise_int32_mul_wraps`.
- **Fix:** do the arithmetic in `std::make_unsigned_t<T>` and cast back.
- Found by UBSan (`build/asan`, see BASELINE.md).

### F-007 `.pyi` stub is badly stale (type checkers and IDEs see a broken API)
- **Status:** open
- `python/pycauset/__init__.pyi` declares 86 names. **71 public names in `__all__` are
  missing**, including `CausalSet`, `field`, `spacetime`, `matrix_rank`, `qr`, `trace`,
  `plot_embedding` and `precision_mode`. It also declares a `MatrixMixin` that does not
  exist. Because a `.pyi` overrides the module for type checkers, mypy and Pylance
  users get "not a known attribute" errors on correct code.
- **Fix direction:** generate the stub (stubgen plus a hand-checked overlay) in CI,
  or delete it until it can be maintained.

### F-008 About 20 documentation examples are broken
- **Status:** open
- Evidence: `review/probes/run_doc_snippets.py` (107 pages OK, 69 fail; about 50 of
  the failures are signature-only blocks that should be fenced as text, not python).
  Real breakage:
  - `pc.MinkowskiDiamond` (top-level alias removed in 9da2970) is still used in
    `docs/classes/field/*` (3 pages), `docs/functions/pycauset.field.md` and
    `docs/classes/spacetime/pycauset.CausalSet.md`.
  - `pc.Matrix` does not exist: `guides/Performance Guide.md` and
    `docs/functions/pycauset.cuda.force_backend.md`.
  - `guides/Numpy Integration.md`: a NumPy ufunc on FloatMatrix fails
    (`__array_ufunc__=None`), contradicting the guide.
  - `docs/classes/SymmetricMatrix.md` and `AntiSymmetricMatrix.md` document item
    assignment that is unsupported.
  - `guides/Storage and Memory.md`: `IdentityMatrix` has no `.properties`.
  - `guides/Matrix Guide.md` block 4: assertion fails.
  - `guides/Examples.md`: uses `AntiDeSitter`, which raises NotImplementedError.
  - `guides/Analyzing a Causal Set.md`: a field on 3D Minkowski raises (no coefficients).
  - `guides/Tutorials.md`: hits the export guard.
  - `guides/Causal Sets.md` block 4: a "1 million element" example allocates about
    125 GB on disk. Running it filled this machine's disk. It must not be a
    copy-paste example.
  - Pages for symbols that don't exist: `pycauset.PersistentObject`,
    `pycauset.CausalSpacetime`, `pycauset.pinning_budget`.

### F-009 Stale test skips hide passing coverage
- **Status:** open
- `test_factorizations.py::test_solve` ("Crashing"),
  `test_lazy_evaluation.py::test_aliasing_safety` ("Segfault investigation") and
  `test_r1_safety_comprehensive.py::test_threaded_io_stress` (Windows-only) all
  **pass** on Linux, 3/3 runs each, with the skip removed. Remove the skips. Run the
  threaded test under ASan/TSan before trusting it.

### F-010 Three C++ test files are orphaned (never built)
- **Status:** fixed: all wired (ctest 9/9) and run in CI (`ci.yml` job `cpp-tests`).
- `tests/test_cpu_worker_contract.cpp`, `tests/test_op_registry.cpp` and
  `tests/test_parallel_utils.cpp` are not in CMakeLists. The first two pass when
  built by hand; the third reveals F-004. CI builds **no** C++ tests at all.

## S3 — minor

### F-011 Several "native" linalg ops always route through `numpy.linalg`
- **Status:** open
- `eigh`, `eigvalsh`, `svdvals` and `matrix_rank` call `numpy.linalg` (full in-RAM
  copy) even though `CpuSolver` implements `eigh`/`eigvalsh`/`svd`. They cannot run
  out-of-core. See `review/probes/numpy_fallback_probe.py`.

### F-012 Most core ops are well below NumPy speed (README claims ≥ 0.90×)
- **Status:** open (S2 for R1 credibility; Phase 6)
- Quiet-machine measurement, `benchmarks/r2_parity.py`, N=1024, 4 vCPU, pip Release
  build: matmul **0.32×**, invert 0.72×, determinant 0.60×, add 0.24×, multiply 0.32×,
  **dot 0.01×** (≈100× slower: 11 ms vs 0.1 ms for a 1M-element dot), solve 1.00×,
  eigh 0.95×. Only 2/8 at parity. `BENCHMARKS.md` claims **8/8** ("R2.2 parity gate"),
  probably measured on the maintainer's machine. Re-measure there before editing the
  claim (PLAN 5.4); a 4-vCPU Linux container may not be representative for BLAS ops,
  but dot at 0.01× cannot be explained by hardware.
- The first session-1 measurement (1.01× matmul, 3× invert) was taken under build load
  and is invalid (BASELINE.md).
- Suspects: the Python wrapper stack (ARCHITECTURE §4.2), per-call allocation and
  temp-file registration, OpenBLAS thread juggling in `CpuSolver::matmul`, and for `dot` a
  non-BLAS path. Profile before changing anything.

### F-013 bool dtype semantics diverge from NumPy without documentation
- **Status:** open. **Decision (maintainer, 2026-11-24, reconfirmed): counting semantics, documented.** bool/bit
  `A + B` adds into integers; matmul counts paths. Document it as a deliberate NumPy
  divergence. With value bounds (features/VALUE_BOUNDS.md) the result is the smallest
  int that fits (bit + bit → int8).
- bool `A + B` → int32 (NumPy: bool OR). bool `A @ B` → int32 (NumPy: bool).
  bool `A * 3` → bool with values unchanged (NumPy: int64 ×3). The last one is wrong
  under any policy.

### F-014 Four thread pools in one process
- **Status:** open
- pycauset `ThreadPool` + OpenMP + pycauset's OpenBLAS + NumPy's bundled OpenBLAS.
  There's no coordination beyond the per-GEMM `openblas_set_num_threads` bump, which
  is itself a process-global race if two Python threads matmul concurrently.

### F-015 Float-scalar ops return `LazyMatrix`; integer ops return concrete matrices
- **Status:** open
- `A * 2.0` gives `LazyMatrix`, which doesn't support item assignment. Type and
  behaviour depend on whether the scalar is a Python float or int.

### F-016 The export guard blocks even tiny disk-backed matrices
- **Status:** open. **Decided: guard by size only** (RESTRUCTURE §3.1 item 7).
- `np.asarray(M)` raises for any file-backed object, regardless of size. A 300×300
  matrix spilled by the governor can't be printed or converted without
  `pc.to_numpy(M, allow_huge=True)`. The spill decision is automatic, so the user
  can't predict when this will happen.

### F-017 Default spill directory is `./.pycauset` in the current working directory
- **Status:** wontfix (maintainer, 2026-11-24): `.pycauset` stays, for full control of
  file lifetime. Document it prominently instead.
- Running a script from a repo or home directory creates spill files there.

## S4 — hygiene

### F-018 Dead code
- **Status:** partly fixed (Tier-A batch 1, session 1; see DEADCODE.md §4 ticks). Remaining:
  tests/manual, tests/stability, debug_load.py, phase3_4_verification.py, mockups/
- `include/pycauset/compute/WorkloadPartitioner.hpp`, `src/compute/WorkloadPartitioner.cpp`
  (unreferenced, not compiled).
- `src/main.cpp` (prints "pycauset Initialized", not compiled).
- `python/pycauset/_internal/lazy_ufunc.py` (never imported), `interop.patch_interop`
  (never called), `ops._streaming_eigh` (unused), unreachable code after `return` at
  `ops.py:1121`, `_storage.StorageRegistry` and `tracked_temp_files` (unused).
- `tests/python/debug_load.py` (Windows CUDA DLL loader script) and
  `tests/python/phase3_4_verification.py` (writes `tmp_verification_phase3_4/` into
  the cwd). Neither is a real test.
- `tests/manual/*` (4 ad-hoc scripts), `mockups/spacetime_studio_mockup.html`.
- CMakeLists: commented-out `causal_tests` and `benchmark_native` targets that point
  at missing files; `CMAKE_UNITY_BUILD OFF` set twice; a "# Touch 2" comment;
  OpenBLAS "Debugging Start" status spam.

### F-019 Debug leftovers in the build
- **Status:** fixed (`#pragma message` removed; `_GNU_SOURCE` guarded; CMake debug spam removed)
- `#pragma message("Spacetime.hpp included")` prints on every compile.
- `_GNU_SOURCE` is redefined in `src/core/MemoryMapper.cpp:1`.

### F-020 Compiler warnings under `-Wall -Wextra`
- **Status:** fixed for `src/` (0 warnings under -Wall -Wextra -Wno-unused-parameter);
  test files still have 3 (they go with the old suite). One was zombie code: a "Send
  Memory Hints" block computed sizes but never sent the hint.
- 18 unique: unused variables in `CpuSolver.cpp` (cpuInfo ×2, out_c, out_d,
  total_bytes, stride_bytes, block_bytes, src ×2), one sign-compare, and
  `-Wclass-memaccess` memcpy into `float16_t` in `bind_matrix.cpp` (×3). The default
  build uses no warning flags, so none of these show up in normal builds.

### F-021 Python lint baseline
- **Status:** open
- `ruff check python/`: 289 findings (149 E501, 95 UP006, 15 UP045, 14 E402, ...).
  Pyflakes on tests, benchmarks, tools, scripts and demos: 87 (62 unused imports,
  21 unused variables).
- 257 broad `except` clauses in the package, 126 of them `except ...: pass`.
- `pyproject` targets py3.8 (`requires-python`, ruff, mypy), but the code uses 3.10+
  idioms and CI only tests 3.12.

### F-022 Doc/plan drift and naming archaeology
- **Status:** open
- `R2 QA` audit references `guides/R2 Feature Menu.md`, which does not exist.
- `TODO.md` says the ThreadPool singleton is leaked; it isn't.
- Code, tests and docs carry milestone names (`R1_*`, `R2_*`, `phase3_4`,
  `phase_b..h`, "Phase 3" comments) instead of describing *what* they are.
- `documentation/dev/Restructure Plan.md` and
  `documentation/internals/plans/archive/Restructure Plan.md` both exist.
- Version: `CMakeLists.txt` has `project(pycauset VERSION 0.1.0)` while the
  CHANGELOG is at 0.6.1.

### F-023 The test harness depends on copying binaries into the source tree
- **Status:** open
- Tests do `sys.path.insert(0, "<repo>/python")` (15 files), so the installed wheel
  is shadowed by the source tree. CI copies `_pycauset.so` and `libpycauset_core.so`
  into `python/pycauset/`. Fix: no path hacks. Test the installed package (or an
  editable install), with one `conftest.py`.

### F-025 `memcpy` from a null pointer when copying an empty object's storage
- **Status:** fixed (`fix(F-006,F-025)`); test `test_F025_save_and_load_empty_matrix`
- `src/core/PersistentObject.cpp:352`: `std::memcpy(dst, src, size)` with
  `src == nullptr` when `size == 0`. UBSan flags it during the persistence tests.
  Guard with `if (size)`.

### F-026 The C++ out-of-core engine (StreamingManager) never runs
- **Status:** open (S2: headline feature unverified)
- `src/compute/StreamingManager.cpp` and `CpuComputeWorker` have **0% line coverage**
  under the full pytest suite plus all demos (DEADCODE.md §3). "Work with matrices
  bigger than your RAM" is PyCauset's headline claim. Either out-of-core works purely
  through mmap paging (and this engine is dead weight to delete), or the tiled path
  is never triggered and untested. Phase 3.5 must answer this with a real
  larger-than-RAM run (cgroup/ulimit-constrained) and decide keep vs delete.

### F-027 pytest output capture swallows sanitizer (UBSan) reports
- **Status:** fixed in CI (`asan.yml`: UBSan with `-fno-sanitize-recover=undefined`, plus `-s`)
- Non-fatal UBSan reports go to fd 2, which pytest captures. Without `-s` a
  sanitized run looks clean even when UB occurs (verified 2026-11-24). `asan.yml`
  passes `-s`, but it runs ASan only (no UBSan) and only on `main`. The new CI
  sanitizer job must use `-s` or `-fno-sanitize-recover=undefined`.

### F-028 Guesswork in the engine (sampled overflow preflight)
- **Status:** resolved by policy clarification (no code change in the end).
- **Policy (maintainer, 2026-11-24, clarified):** *guesses may inform, never decide.*
  Sampling and heuristics may produce warnings, advice and diagnostics, or power
  explicitly statistical operations. They may never change a result's values, dtype or
  success/failure.
- History: the sampled int32 matmul preflight (`maybe_warn_integer_matmul_overflow_risk_preflight`,
  `bind_matrix.cpp`) was removed under the first, stricter reading, then **restored**
  once clarified, because it only warns. It's useful: an early warning before a long
  computation that would end in `OverflowError`. Now labelled advisory in code and
  message; the garbled "≈" in the message was fixed.
- Audit result: this is the only sampling heuristic in the engine. Byte and threshold
  estimates are deterministic arithmetic on shapes. The `AutoSolver` CPU/GPU timing
  benchmark picks a device, not a result (GPU-only; it leaves with CUDA).
- Verified: float reductions are bit-identical across runs; overflow detection is exact
  whether or not the advisory fires (`test_F028_*`).
- Related inconsistency: int32 **matmul raises** on overflow, while int32 **elementwise
  ops wrap** (F-006). One overflow policy is needed; it arrives with value bounds.

### F-029 The old test suite imports an arbitrary copy of pycauset
- **Status:** fixed (pyproject `pythonpath = ["python"]`)
- pytest puts the repo root on `sys.path` (because `tests/python/__init__.py` exists), and
  only some test files insert `python/`. The **first file to import pycauset decides**
  which copy every test uses. In practice that was the pip-installed copy in
  site-packages, so local runs silently tested a **stale build**, not the source tree.
  Found when a fixed bug (F-024) still reproduced in the old suite. CI was unaffected
  (it installs and copies the same build).
- Fix: `[tool.pytest.ini_options] pythonpath = ["python"]`. The per-file `sys.path` hacks
  are now redundant and go away with the old suite (TESTS.md §4).

### F-030 `pc.eigvals_arnoldi` never uses the C++ Arnoldi solver; the solver is unrestarted
- **Status:** open (S2; maintainer needs Arnoldi for R1). Diagnosed session 1.
- The public function computes **all** eigenvalues densely with NumPy (exact, but in RAM
  and O(n³)) because a code comment says the native solver "crashes with an access
  violation". **The crash no longer reproduces** (n = 50/200/500; likely fixed with F-004/F-005).
- **Real problem:** `CpuSolver::eigvals_arnoldi` is a *single* Arnoldi pass of `m` steps
  with **no restarts and no convergence check**. `tol` is only a breakdown threshold, so
  it silently returns unconverged Ritz values: relative error 3e-5 … 2.5e-3 vs NumPy with
  `tol=1e-10` on random symmetric matrices. Routing users to it now would violate
  "guesses may inform, never decide".
- **Fix plan (tech lead):** implement a restarted method with residual-based convergence
  (`‖A x − θ x‖ ≤ tol·|θ|` per wanted Ritz pair) that **raises** on non-convergence (like
  SciPy's ArpackNoConvergence). Options: (a) vendor **Spectra** (header-only, MPL-2.0,
  ARPACK-style implicit restarts on Eigen, which we already use); preferred, but GitHub
  archive downloads are blocked in the review sandbox, so do it where the network
  allows; (b) hand-written Krylov–Schur (general) plus thick-restart Lanczos (symmetric).
  Then route `pc.eigvals_arnoldi` to native, delete the stale crash comment, and test
  against NumPy on symmetric, non-symmetric and skew-symmetric inputs, including clustered
  spectra.

### F-031 `tests/test_symmetric_matrix.cpp` asserts nothing in release builds
- **Status:** open (S3)
- It uses plain `assert()` (no gtest macros). CMake builds the tests as RelWithDebInfo,
  which defines NDEBUG, so every check is compiled out and the test always "passes".
  Convert it to gtest `EXPECT_*` (or fold it into the new suite).

### F-032 Unregistered template instantiations reach Python as bare `MatrixBase`
- **Status:** open (S2; blocks the "every dtype works with every structure" gate, PLAN 4.1d)
- The C++ factory creates `TriangularMatrix<T>`, `DiagonalMatrix<T>`, `IdentityMatrix<T>`
  for ~10 dtypes each, but Python classes exist only for Triangular{bit,int32,float64},
  Diagonal{float64} and Identity{float64}. Any op that yields another instantiation
  (e.g. `C * np.int64(3)` before the ScalarOps guard) returns a bare `MatrixBase` with
  almost no methods (no `get_data_type`, no export).
- Interim: ScalarOps produces triangular results only for int32/float64. Fix: register
  every instantiation the factory can create (generated binding table), or narrow the
  factory to the registered set. Add an invariant test: every result's Python type is a
  concrete registered class.

### F-033 NumPy export: only 3 of 13 dense dtypes had a native path; 64-bit ints lost precision
- **Status:** fixed (`fix(F-001,F-002,F-003)` batch)
- `_to_numpy_fast` handled only float64/int32/complex128. All other dense dtypes fell back
  to a per-element Python loop that read values through `get_element_as_double`, so
  int64/uint64 values above 2^53 lost precision, and uint64 values near 2^64 raised
  "Python int too large to convert to C long" (the "uint64 a-b" error in the first probe).
- Fix: one generic native exporter for every dense dtype (`export_dense_any` in
  bind_matrix.cpp, replacing ~150 duplicated lines); the Python fallback reads integers
  through the exact typed `get`. Also asserts the F-001 invariant (no lazy scalar on ints).
