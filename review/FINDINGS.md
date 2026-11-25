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
- **Status:** largely fixed (session 6). Root causes: RAM allocation zero-filled every page
  (F-057), views read per element (F-043), matvec had no BLAS path. 3000² float64, 4 vCPU
  (pycauset vs NumPy): A+B 5.7 vs 16.4 ms, A*B 6.8 vs 20.4, 2A+3B 13.6 vs 41.3, A.T+B 11.2 vs
  23.7, A+1.5 5.8 vs 10.7, A@B 168 vs 179, A@v 1.6 vs 1.0–6.5 ms (gemv).
- Session 7 (`benchmarks/r2_parity.py`, N=1024, best of 9, ±10% machine noise): matmul 0.87–0.92,
  solve 0.91–0.93, invert 0.7–1.1, dot 1.0–1.8, add 0.9, multiply 0.8, **determinant 0.68**
  (open), eigh 0.89–0.98, export 0.9–1.0. Three causes fixed:
  * **Thread-pool spinning** (the dominant one): OpenMP, our OpenBLAS and NumPy's bundled
    OpenBLAS share the cores and each spins after a call (OpenBLAS ~2^28 cycles ≈ 0.1 s). The
    pool that runs next is starved: pycauset matmul 24 ms right after a NumPy call vs 8 ms
    alone; NumPy's own matmul 1.7× slower next to us; a 1M dot 1–4 ms vs 0.1. `import pycauset`
    now sets `OMP_WAIT_POLICY=PASSIVE` and `OPENBLAS_THREAD_TIMEOUT=4` defaults (user values
    win) before any BLAS loads; NumPy's OpenBLAS is covered only when pycauset is imported
    first (or the user sets the variable). Documented in the performance guide.
  * Export: `np.asarray` 250 ms → 0.56 ms for 1024² (`export_generic`, one copy, F-048 family).
  * dot: BLAS `ddot`/`dsdot` instead of an OpenMP reduction.
  The benchmark used to time a NumPy export inside every op (now its own "export" row) and
  imported NumPy first.
- Original measurement:
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
- **Status:** resolved (session 3): **live, now tested.** It runs only when an operation
  exceeds the governor's RAM budget (real free RAM minus a safety margin); the
  user-facing `set_memory_threshold` doesn't affect that decision, so tests never reached
  it. Internal helpers `_get/_set_ram_safety_margin` plus a `streaming.*` kernel trace
  now let `tests/engine/test_streaming.py` force and verify it (54 cases). Findings from
  that: F-040.
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
- **Status:** fixed (`fix(F-030)`, session 3): `CpuSolver::eigvals_arnoldi` now uses
  **Spectra** (header-only, MPL-2.0; system `libspectra-dev` or fetched v1.0.1): implicitly
  restarted Lanczos for symmetric input (float64 output), Arnoldi otherwise (complex128),
  matvecs through `CpuSolver::matrix_vector_multiply` (any structure, disk-backed), raises
  on non-convergence, deterministic start vector, dense fallback for tiny problems. Public
  `pc.eigvals_arnoldi` routes to it; the NumPy fallback, the stale crash comment and the
  Python "streaming" variant (which materialized the matrix) are deleted. Accuracy vs NumPy
  now ~1e-14 (was ~1e-3). Tests: `tests/engine/test_eigensolvers.py` (also covers
  `eigvals_skew`/`eig_skew` on causal-derived matrices).
- Original diagnosis, kept for history:
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
- **Status:** fixed by construction (session 4): one Python class per structure; a pybind11
  polymorphic type hook (`src/bindings/bindings_common.hpp`) maps every instantiation to its
  structure class. ScalarOps' int32/float64-only triangular restriction can now be lifted (4.1d).
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

### F-034 int ⊕ float under-promoted across kinds: silent overflow / precision loss
- **Status:** fixed (`fix(F-034)`)
- `int32(100000) + float16(0)` returned **inf**, no warning: the resolver used the float
  operand's dtype even when it can't hold the integer's values. Integer division also
  returned float32 for int32/int64 (lossy above 2^24), while the code comment said float64.
- The documented policy already said under-promotion applies *within* a kind, not
  across kinds; the code didn't follow it. Fix (`PromotionResolver.cpp`): the cross-kind
  result float must hold the integer dtype (bit/int8/uint8 → float16, 16-bit → float32,
  32/64-bit → float64); integer division → max(float32, that). Within-float
  under-promotion (float32 ⊕ float64 → float32) is unchanged. Value bounds (F1) will later
  allow narrower floats when the values provably fit.
- Found by the dtype-pair survey (980 combinations). Same survey, still open: mixed-dtype
  matmul with float16/complex operands has no kernel (64 combos raise); signed int ⊕
  uint64 raises (NumPy gives float64; 32 combos). See F-035.

### F-035 Mixed-dtype gaps found by the dtype-pair survey
- **Status:** (a) fixed (`fix(F-035a)`): `dispatch_matmul` promotes mixed float16/complex
  operands to the resolved dtype via `pycauset::cast_to` (72 grid cases now pass; an old
  self-skipping block-matrix test now runs). (b) open.
- (a) `A @ B` where one operand is float16 or complex and the dtypes differ raises
  "CpuSolver::matmul currently only supports DenseMatrix<double> result for generic inputs"
  (64 combinations, e.g. int32 @ complex64, float32 @ complex128, int8 @ float16).
  Fix: promote (cast) operands to the resolved result dtype, then use the same-dtype kernel.
- (b) Any signed int ⊕ uint64 raises "unsupported signed/unsigned mix" (32 combinations).
  NumPy gives float64, which silently loses precision above 2^53. **Decision (tech lead,
  session 2): keep raising** (loud beats silently approximate, README rule 12). Resolve
  with value bounds (F1): when the uint64 operand's bounds fit int64, the exact result is int64.

### F-036 Identity-matmul shortcut bypassed dtype rules (and F-001's invariant)
- **Status:** fixed (`fix(F-035a)`)
- `dispatch_matmul`'s "Identity × A → A * scalar" shortcut called the class's
  `multiply_scalar` directly: for integer A it created a lazy-scalar view (the F-001 bug
  class) and the result dtype ignored the resolver (float64 identity @ int32 stayed int32).
  Now routed through `scalar_op` + `cast_to` with the resolver's dtype.
- Knock-on: `matrix_power` seeded its loop with a float64 identity, so causal powers became
  float once dtypes were respected. The loop no longer uses an identity seed (which also
  saves one matmul); `matrix_power(C, k)` stays `TriangularIntegerMatrix`.

### F-037 save → load silently un-transposed matrices (data corruption)
- **Status:** fixed (`fix(F-037)`)
- `pc.save(A.T, p); pc.load(p)` returned **A** (wrong shape, wrong data), with no error.
  Save records the logical shape plus `is_transposed`; load passed that shape to
  `_from_storage`, which expects the *storage* shape and applies the transpose again.
- Fix in `persistence.load`: derive the storage shape (swap for transposed views) for the 16
  rectangular constructors. Fixing it on load means already-saved files load correctly too.
- Found by `tests/engine/test_invariants.py` (save/load round-trip over the object catalog).
  Regression: `test_regressions.py::test_F037_*`.

### F-038 `LazyMatrix` is a second-class object
- **Status:** fixed for users (`fix(F-038,F-039)`, session 3): operators evaluate eagerly. Measured
  first: lazy `(A+B)*2-A` on 2000² was 100.8 ms vs 91.9 ms eager (NumPy 7.1 ms), so laziness
  bought nothing, and it was float-only. Tech-lead decision: remove lazy evaluation from R1
  operators; the expression engine itself is deleted in a follow-up commit. Any future fused
  kernels get designed in the performance phase.
- For float matrices most element-wise ops return a lazy expression (`A + B`, `A - B`,
  `A / B`, every scalar op, `(A + B) * 2.0`), but `A * B` returns a concrete matrix.
  `LazyMatrix` is not a `MatrixBase`: `pc.sum` rejects it, it has no `.T`/conj/properties,
  and `pc.save` fails ("missing copy_storage"). Users hit this constantly with ordinary
  arithmetic. (Concrete form of F-015.)
- Fix direction (tech lead): keep lazy fusion (valuable out-of-core), but make the lazy
  object a full citizen: every API that needs a concrete matrix materializes it
  automatically (one C++ `materialize()` used by the operator table, `sum`, save, views),
  and `A * B` follows the same rule as its siblings. Decide with the operator-table
  consolidation (PLAN 4.3). Tests waiting: invariants marked `F-038`.

### F-039 Triangular + transposed triangular silently dropped half the matrix
- **Status:** fixed (`fix(F-038,F-039)`)
- `C + C.T` (the comparability matrix, a staple of causal-set analysis) and `C - C.T`
  returned a **TriangularIntegerMatrix with the lower half lost**, silently. `C * C.T` raised.
  Pre-existing: integer/bit ops never used the lazy path.
- Root cause: two structure resolvers with different rule sets (`MatrixTypeResolver.hpp`,
  eager; `StructureResolver.hpp`, lazy only), and neither knew orientation: a transposed
  triangle is a *lower* triangle.
- Fix: one op-, orientation- and strictness-aware resolver
  (`structure::resolve_result_structure`, `src/core/StructureResolver.cpp`) used by the eager
  add/subtract/elementwise_multiply; it only picks structured classes that exist for the
  result dtype (F-032), else dense. `MatrixTypeResolver.hpp` deleted. Diagonal/identity/
  symmetric structure propagation (previously lazy-only) now works on the eager path.
- Tests: `test_regressions.py::test_F039_*` (causal and float triangles, + − *).

### F-040 Out-of-core streaming ignored view metadata (silent wrong results)
- **Status:** fixed (`fix(F-026,F-040)`)
- `StreamingManager` copies tiles straight from raw row-major storage, ignoring
  transpose, slice offsets, conjugation and the lazy float scalar. Under real memory
  pressure, `A.T @ B` returned completely wrong values (100% mismatch in the forced test)
  and `(A * 2.5) @ B` would have dropped the 2.5. Complex and mixed-dtype operands threw
  mid-operation ("only supports Float32/Float64").
- Fix (`CpuDevice.cpp`): stream only plain dense same-dtype operands of supported dtypes
  (matmul: float32/64; element-wise: also int32/64); everything else takes the direct
  kernels, which honour views and still work out of core via mmap paging.
- Open follow-ups: (a) teach the tiler to read views (transposed/offset tiles) so views
  stream too; (b) streaming for complex/other dtypes; (c) `pc.matrix(ndarray)` raises
  "Import rejected by MemoryGovernor" when the array exceeds the RAM budget, rather than
  writing straight to disk; (d) the streaming decision ignores `set_memory_threshold`.
  Decide (d) as part of the storage review (PLAN 3.1/3.5).

### F-041 Python-side second-class objects and dtype guessing
- **Status:** (a) fixed; (b) fixed (session 6, maintainer decision 2026-11-25): float64 default
  like NumPy, documented, `dtype=` overrides; `_internal/lazy_allocation.py` deleted.
- (a) `pc.TriangularMatrix(n)` built a pure-Python, list-of-lists toy matrix (no `@`, outside
  the engine), contradicting its own docs ("abstract base class"). It's now abstract:
  `isinstance(native_triangular, pc.TriangularMatrix)` still works; constructing it raises
  a TypeError pointing to the real constructors. ~60 never-executed lines deleted from
  `_internal/matrix_api.py`.
- (b) `pc.zeros/ones/empty(shape)` without `dtype` return a dtype-deferred wrapper
  (`_internal/lazy_allocation.py`, 200 lines) that decides its dtype later: int32 by default,
  or from the first value written. That's a second-class object (like LazyMatrix) and a
  guessed dtype, and NumPy's default is float64. It was a deliberate R1 feature
  (CHANGELOG 0.6.1), so it's the maintainer's call. **Tech-lead recommendation:** default
  to float64 like NumPy and delete the wrapper.

### F-042 Scalar factor views exist only for float dtypes (wanted feature, R1)
- **Status:** in progress. **Done (session 5):** dense and triangular integer matrices — `k*A`
  and `-A` are O(1) shared-storage views with an exact folded integer factor (|k| <= 2^53,
  folded mod 2^64, NumPy wrapping on read). 3000² int32 `5*(3*(2*A))`: 85 ms -> 0.21 ms.
  Every consumer verified by `tests/engine/test_int_scale_views.py` (7 dtypes x 4 factors x
  17 consumers + triangular + mixed). Symmetric/antisymmetric integer views added (same tests).
  **Remaining:** bit matrices (`2*C` needs a cast view: logical dtype != storage dtype).
  Not planned: vector and diagonal views (O(n) copies; not worth the reader surface, LAZY.md).
- `k * M` is meant to be O(1): the result shares `M`'s storage and records `k` in the view
  metadata (`scalar_`), and chains such as `2 * 3 * 5 * M` fold into one factor. Measured on
  3000×3000, three chained multiplies: float64 0.2 ms (view, `scalar = 30`); int32 85 ms and
  int64 139 ms (three full copies).
- History: before F-001 the integer path also made a view, but most readers (`__getitem__`,
  NumPy export, matmul) ignored `scalar_` for integer and bit matrices, so `A * 3` returned
  `A` unchanged. The F-001 fix made integer and bit scaling eager and exact, which is correct
  but lost the O(1) path.
- Not the same thing as the expression engine removed in `d564c7f` (fused `A + B + C` trees,
  float-only, measured no faster). The view mechanism (`scalar_`, `is_transposed_`,
  `is_conjugated_`) stays and is the right place for this.
- Plan:
  1. Store the factor exactly: an int64 factor for integer/bit views beside the complex one
     for floats. Wrapping multiplication is a ring homomorphism mod 2^k, so
     `(x·a)·b ≡ x·(a·b) (mod 2^k)`: folding int factors gives bit-identical results to eager
     NumPy, overflow included.
  2. Every reader honours the factor (element get, export, sum, matmul via BLAS `alpha`,
     element-wise kernels, streaming, persistence save/load), enforced by a conformance
     test that runs each op on `k * M` and compares it with NumPy (the gap that hid F-001).
  3. Bit matrices: `2 * C` is an int view over bit storage (dtype from the promotion rules);
     an int32 view costs 1/32 of the memory of the eager result.
  4. Cases that can't be a view stay eager: int × float scalar (truncation per element),
     dtype-changing scalars.

### F-043 Reading a scaled view costs 2× (views aren't free when consumed)
- **Status:** fixed for element-wise ops (session 5): a strided row reader applies transpose,
  offsets, broadcasting, factor and conjugation in the kernel's inner loop (CpuSolver.cpp
  `StridedReader`). 3000² float64: `2A+3B` 64 → 37 ms (A+B 31 ms), `A.T+B` 75 → 36 ms
  (NumPy 27 ms). Matmul already absorbed the factor via BLAS alpha.
- 3000² float64, 4 cores: `A + B` 32 ms, `(2*A) + (3*B)` 62 ms. The scaled operands leave the
  fast kernel path. A view must be free on read (the factor belongs in the kernel's inner
  loop or in BLAS `alpha`), otherwise laziness trades a copy now for a slowdown later.

### F-044 `.T` copies Diagonal and Symmetric matrices
- **Status:** fixed (session 5). Symmetric/antisymmetric `.T` already shared storage (the probe's
  "copy" flag was a false reading); Diagonal `.T` copied (27 ms for 5M) and is now an O(1) shared
  clone.
- Both are their own transpose; `.T` should return the same storage in O(1). Today it
  allocates and copies (probe: `review/probes/view_semantics_probe.py`).

### F-045 `.get()` and writes ignored view metadata (scale factor, conjugation)
- **Status:** fixed (structure-class commit). Tests: old suite + engine suite.
- `B = A * 2.5; B.get(0, 1)` returned the unscaled value (2.0) while `B[0, 1]` gave 5.0; on a
  conjugated complex view `.get` returned the unconjugated value. Writes were wrong too:
  `B[0, 0] = 1.0` read back as 2.5, and writes into a conjugated view read back conjugated.
  Silent wrong data. Cause: the per-class bindings bound the raw typed `get`/`set`, which apply
  the transpose but not the factor or conjugation.
- Fix: reads go through the view-aware readers (floats/complex) or exact typed reads (ints/bits);
  writes first fold a scale/conjugation view into private storage (dense). Structured matrices
  carrying a float factor raise on write instead of storing a wrong value; lazy phase L1
  (features/LAZY.md) replaces this with ViewSpec-aware writes.

### F-046 `pycauset.matrix(n)` silently used float32 for n >= 10,000; list input inferred structure
- **Status:** fixed (structure-class commit).
- A size heuristic changed the result dtype ("Enforcing Float32 precision for storage efficiency"),
  violating "guesses may inform, never decide". A nested list whose values happened to be strictly
  upper triangular became a triangular class, while the same data as a NumPy array became dense.
  Now: dtype from `dtype=` or the data (NumPy rules, Python ints -> int32); lists take the array path.
- Also fixed on the way: persistence recorded every diagonal/symmetric/antisymmetric/identity
  matrix as float64 whatever its dtype (wrong dtype after load); files now store structure + dtype.

### F-047 Eigen-decompositions don't follow under-promotion (float32 in -> float64 out)
- **Status:** fixed (session 6). eig/eigvals/eigh/eigvalsh compute in double and return
  float32/complex64 for float32/complex64 input (NumPy); other inputs (float16, ints, bits) are
  converted to double first, which also fixed `A.eigh()` raising "not implemented" for float32.
- `pc.eigvalsh(A)` / `A.eigvalsh()` on a float32 matrix return float64 eigenvalues (NumPy returns
  float32). The removed per-class float32 method returned float32, so behaviour differed by entry
  point. Decide in the operator consolidation (4.3): result dtype = input float dtype.

### F-048 Tests and the doc runner imported a stale installed pycauset
- **Status:** fixed. A subprocess-based test and `review/probes/run_doc_snippets.py` imported the
  site-packages copy (from an earlier `pip install`), so they tested old code (same class as
  F-029). Both now put the source tree on PYTHONPATH; the stale install was removed from the
  dev container. Rule (TESTS.md): never rely on an installed pycauset in tests or probes.

### F-049 Element-wise ops on integer views went through double (int64 precision loss)
- **Status:** fixed (with F-043). Any view operand (transpose, slice) took a per-element path that
  read integers via `get_element_as_double`: `A.T + B` on int64 values near 2^62 was off by 1.
  Now read exactly in the operand's own type. Test: `tests/engine/test_views.py` (int64/uint64
  values beyond 2^53, every view kind).

### F-050 Conjugation and transposition lost in several view paths (silent wrong results)
- **Status:** fixed (session 5). Found by the new view × structure × scalar-op grid
  (`test_views.py`, 326 cases vs NumPy):
  - the element-wise fast path and the scalar-op fast path didn't check `is_conjugated`:
    `Z.conj() + Z` returned `2·Z` instead of `2·Re Z`;
  - `DenseMatrix::add_scalar` ignored conjugation and re-applied the transpose flag to an
    already-transposed result: float `A.T + 1.5` returned the untransposed matrix (wrong shape);
  - `SymmetricMatrix`/`AntiSymmetricMatrix::clone` dropped the view metadata, so for an
    antisymmetric A: `A.T * 3 == 3A`, `-A.T == -A`, `A.T.T == -A`;
  - `SymmetricMatrix + s` was unimplemented (raised); now symmetric, and antisymmetric + s dense.
- Complex products differ from NumPy by <= 1 ulp (multiplication formula, not views): noted, tested
  with a 1-ulp tolerance.

### F-051 Integer matvec / vecmat / dot / sum went through double (int64 precision, UB on overflow)
- **Status: fixed** (sessions 5 and 7).
- Also fixed (session 5): element-wise ops with a non-dense integer operand (symmetric, triangular,
  diagonal) or a structured integer result read through double; now exact (`exact_int_reader`).
- **Status:** matvec and vecmat fixed (session 5: exact integer path, wrapping 64-bit accumulation,
  `cpu.matvec.int_exact` / `cpu.vecmat.int_exact`). Integer dot fixed (session 7): `pc.dot` and
  `v @ u` share `vector_dot` (`bind_vector.cpp`): integer/bit operands give an exact Python `int`
  (128-bit accumulation, integer view factors applied, `OverflowError` past 127 bits; bit·bit
  keeps the popcount kernel). Before, `pc.dot` returned a float and `v @ u` for int32/int16
  **ignored the vector's factor** (55 instead of 165 for a factor-3 view). `pc.sum` fixed
  (session 7, `math/Reductions.cpp`): NumPy result kinds (was always `complex`), exact integer
  sums, deterministic float sums (row order, not OpenMP reduction order), popcount for bit
  matrices (|C| at n = 20,000 in 2 ms; was ~4e8 virtual calls) and O(1) for C ± Cᵀ closures.
  Cached integer sums are dropped (not rescaled) when a factor changes: elements wrap.
- Was: `M @ v` with int64 entries near 2**62 off by ~57; out-of-range double->int conversion in
  the fallback is undefined behaviour. Test: `tests/engine/test_int_scale_views.py` (matvec, vecmat).

### F-052 Eager scalar ops on symmetric integer matrices returned dense (lazy/eager class mismatch)
- **Status:** fixed (session 5), found by the new `PYCAUSET_LAZY=0` oracle run of the engine suite:
  `3 * S` (int symmetric) was a SymmetricMatrix lazily but a DenseMatrix eagerly. The exact scalar
  kernel now keeps symmetric structure (any op) and antisymmetric structure (zero-preserving ops).
  CI runs the engine suite in both modes.

### F-053 `CausalSet.links()` reported false links (uint8 path counts wrapped)
- **Status:** fixed (session 5). `links()` computed C² as a NumPy uint8 product: 256 paths wrapped to
  0, so pairs with 256 intermediate elements counted as links (a 300-chain had 342 "links", not 299).
  Now a native transitive reduction on the bits (`_transitive_reduction`, valid because a CausalSet's
  matrix is validated transitive): n = 20,000 in 0.05 s, no n² integer product. The NumPy export of
  the result dominates `links()` time for mid n (non-dense export is slow; F-012 family).

### F-054 Bit `&`/`|`/`^` lost their word-level fast path (175× slower)
- **Status:** fixed (session 6). The structure-class rewrite replaced the per-class bit operators
  with a per-element loop: n = 2000 `A ^ B` 71 ms; restored word-level path: 0.4 ms.

### F-055 Integer matmul has no fast path (int16 matmul 54× slower than float64 BLAS)
- **Status:** fixed (session 6) without waiting for value bounds: the bound k·max|A|·max|B| < 2^53 is
  MEASURED on the operands (O(n²) next to the O(n³) product) and the product runs exactly through
  float64 dgemm (`cpu.matmul.int_via_dgemm`); otherwise the exact integer kernels. n = 2000 int16:
  5.9 s -> 0.12 s. Overflow of the output dtype still raises. Value bounds (F1) will skip the scan. n = 2000: int16 @ int16 5.9 s, float64 0.11 s,
  bit @ bit 0.05 s. Fix: when value bounds prove every partial sum < 2^53, run integer matmul
  exactly through float64 BLAS (and narrow back). See features/BIT_PLANES.md.

### F-056 `pycauset.vector(float32 array)` silently became float64; complex 1-D import missing
- **Status:** fixed (session 6). A helper promoted float32 vectors to float64 (float16 and every
  other dtype were kept), so float32 eigenvalues came back as float64 through the NumPy path
  (part of F-047). Complex 1-D arrays are now imported natively.

### F-057 RAM allocation zero-filled and pre-faulted every page; "discard" zeroed RAM data
- **Status:** fixed (session 6). Linux RAM storage was a memfd mapped MAP_SHARED|MAP_POPULATE:
  allocating a 3000² float64 result took 23 ms (NumPy ~8 ms) and dominated element-wise ops
  (F-012). Now private anonymous memory (as malloc / the Windows VirtualAlloc path) with
  transparent huge pages: `A + B` 35 ms -> 5.7 ms (NumPy 16.4 ms, single-threaded here).
- The switch exposed a latent data-loss bug: the I/O "discard" hint (`madvise(MADV_DONTNEED)`,
  Windows `OfferVirtualMemory`) on RAM-backed storage frees the pages, so data reads back as
  zeros (block matrices lost their blocks). Discard is now a no-op for RAM storage on all
  platforms; on Windows this was already reachable before this change.

### F-058 Vector arithmetic was a second, weaker implementation; matrices lacked `s / M`
- **Status:** fixed (session 7).
- Vectors had no `/`, `v - s`, `s - v`, unary `-`, element-wise `*` or `/` (TypeError), and
  their scalar ops turned int8/int64/uint* vectors into float64 through `double` (int64 values
  beyond 2^53 lost precision). Matrices had no scalar `__rtruediv__` (`2 / A` TypeError for
  every dtype; found by the vector oracle test).
- Fix: a dense vector is a 1 x n matrix over the same storage (bit vectors now padded to 64
  bytes like a bit-matrix row), so vector arithmetic runs through the matrix machinery
  (`math/VectorOps.hpp`) with the matrix dtype rules. Matrix operators collapsed from ~45
  pybind overloads to one `matrix_arith` path (PLAN 4.3). Deleted: `add_vectors`,
  `subtract_vectors`, `scalar_multiply_vector`, `scalar_add_vector` and their device/solver
  kernels (~1,200 lines), `VectorBase::multiply_scalar/add_scalar`.
- Speed (1M elements): scalar kernels now read contiguous storage directly and run in parallel
  for vector-shaped results (`2*v` 6.4 -> 1.2 ms, NumPy ~0.8; int `3*v` 4.9 -> 0.8 ms).
- Test: `tests/engine/test_vector_ops.py` (vector op == row-matrix op == NumPy values).


### F-059 Complex trace/determinant dropped the imaginary part; triangular det was 0 for most dtypes
- **Status:** fixed (session 7): `determinant` returns complex through the device chain
  (no tolerance), triangular uses `TriangularMatrixBase::has_diagonal()`, LAPACK called
  column-major (no transpose copy); `trace` in `math/Reductions.cpp` (exact int / float /
  complex); the Python wrappers no longer swallow native errors into a NumPy fallback.
- `pc.trace` and `pc.determinant` returned `double` all the way down (device interface), so a
  complex matrix gave only the real part: det([[1+2j, 3], [0, 2]]) = 2.0 (NumPy 2+4j); every
  complex trace lost its imaginary part. The complex-vs-real choice also used a tolerance on
  the scalar's imaginary part (`> 1e-14`): guesswork deciding a result.
- Triangular matrices with a diagonal: `determinant` looked up `has_diagonal()` only on the
  float64 and int32 classes, so float32/int16/int64/uint8/... gave det = 0 (should be the
  diagonal product).
- Integer trace went through double (int64 precision).
- Plan: trace into `math/Reductions.cpp` (exact int / float / complex); determinant returns
  complex for complex dtypes; `TriangularMatrixBase::has_diagonal()`; LAPACK called
  column-major on the row-major buffer (det(Aᵀ) = det(A)) to skip LAPACKE's transpose copy.

### F-060 `A[1:4, 1:4] @ B` raised ("views with offsets not supported")
- **Status:** fixed (session 7): `dispatch_matmul` writes windows out first. NumPy supports it; slices should be written out (or read
  strided) by matmul itself. `inverse`, `qr`, `lu` accept slices.

### F-061 Matmul ignored conjugation: `A.H @ B`, `A.conj() @ B` were wrong for complex matrices
- **Status:** fixed (session 7; found by a linear-algebra × view probe vs NumPy).
- The complex matmul kernels read raw storage (transposition and the scalar factor are
  handled; the conjugation flag is not). `A.H @ B` (the Hermitian product used for
  Pauli-Jordan / Wightman constructions) computed `A.T @ B`. Also wrong through it:
  `matrix_power` and `pinv` of conjugated views.
- Fix: `dispatch_matmul` writes conjugated (and windowed, F-060) operands out before the
  kernels; an O(n^2) copy in front of O(n^3) work.

### F-062 Linear-algebra kernels ignored view factors and windows (`eigvals(2*A)`, slices)
- **Status:** fixed (session 7). Test: `tests/engine/test_linalg_views.py` (10 ops × 7 view
  kinds × 4 dtypes vs NumPy). `eigvals(2 * A)` for float64 returned A's eigenvalues (the
  O(1) scale view's factor was dropped). Zero-offset slices (`A[0:3, 0:3]`) were read as the
  first 9 storage elements in `np.asarray`, `solve`, `svdvals`, `inv` (int path), ...:
  `has_view_offset()` is false for a window that starts at (0, 0) but is narrower than its
  storage, and ~25 call sites used it as "plain storage" (two had local patches).
- Fix: `MatrixBase::is_subview()` (offset OR logical dims != storage dims) replaces
  `has_view_offset()` everywhere; linear-algebra entry points (eig*, svd, solve, det, inverse,
  qr, lu, cholesky) write any view (factor, conjugation, transpose, window) out before the
  kernel, so kernels only ever see plain storage.

### F-063 Smaller linear-algebra defects found by the same probe
- `pc.invert` of a complex matrix returned float64 with the imaginary part dropped (the
  Python inverse wrapper cast NumPy's fallback result to float64). **Fixed** (session 7).
- `pinv` used Aᵀ instead of Aᴴ: wrong for rectangular complex matrices. **Fixed.**
- Diagonal inverse: entries with |x| < 1e-12 became 0 instead of 1/x (a tolerance deciding a
  result; an exact 0 should raise). **Fixed** (exact 0 raises).

### F-064 `conj()` / `.H` of a triangular matrix with a diagonal read shifted values
- **Status:** fixed (session 8). Found by the operation atlas on its first run (Phase C).
  `TriangularMatrix(has_diagonal=True).conj()` exported the strict upper triangle shifted
  by one and lost the diagonal (`[[1,3],[0,4]]` became `[[0,1],[0,0]]`); every operation on
  the conjugated view (add, export, inv, ...) was wrong, silently, for every dtype.
- Cause: `TriangularMatrix<T>` had no `clone()`; the generic `MatrixBase::clone` rebuilds
  through `ObjectFactory`, which does not know `has_diagonal`, so the packed row offsets
  were computed for a strict triangle. Fix: a `clone()` override that keeps it (as
  `SymmetricMatrix` already had). Test: the atlas cells `*/triangular/conj|H`.

### F-065 Antisymmetric matrices with unsigned or bool dtype gave contradictory values
- **Status:** fixed (session 8, atlas). `AntiSymmetricMatrix(dtype="uint8")` stored
  `A[1,0] = 254` for `A[0,1] = 2` (wrapped), while `A * 2.5` gave `-5.0` and bool `B + B`
  gave `-2` where the export said `True`. There is no consistent meaning for A(j,i) = -A(i,j)
  in unsigned or bit storage. Decision: antisymmetric requires a signed dtype; the
  constructor raises `ValueError` for unsigned and bool (pre-alpha: no compatibility path).

### F-066 Triangular matmul: diagonal dropped (int32), transposed operand ignored (all)
- **Status:** fixed (session 8, atlas). `T @ T` for an int32 `TriangularMatrix` with a
  diagonal gave only the strict part (`cpu.matmul.tri_i32` summed k in (i, j) and allocated
  a strict result). `T @ T.T` (upper x lower, not triangular) was computed as `T @ T` into a
  triangular result for float64, and `C @ C.T` for a causal matrix likewise: the triangular
  kernels read upper storage and ignore the transpose flag.
- Fix: `dispatch_matmul` writes a transposed triangular operand out (dense) before the
  structured paths, so the kernels only see upper storage; the int32 kernel sums k in [i, j]
  and its result keeps a diagonal iff both operands have one.
- Performance part resolved by F-080: one templated triangular kernel reads packed storage
  for every real dtype (`cpu.matmul.tri_i32`, which used the virtual accessors, is gone).

### F-067 `matrix_power(2*I, n)` returned `2*I`
- **Status:** fixed. The Python `matrix_power` returned an identity operand unchanged
  (ignoring its scale factor, for every n including negative). The shortcut is removed:
  identity x identity is O(1) in the engine anyway.

### F-068 `solve`: vectors always went to NumPy; structured operands wrong; `solve_triangular` broken
- **Status:** fixed (session 8, atlas).
  * The native `solve` took matrices only, so `solve(A, vector)` (the common call) raised
    inside and the Python wrapper silently retried in NumPy, for every input.
  * The native kernel handled dense float32/float64 only (int, complex, structured raised →
    NumPy), and the Python wrapper's own shortcuts were wrong: an identity operand returned
    `b` unscaled; diagonal/triangular operands went to `solve_triangular`, whose diagonal
    path dropped imaginary parts and broadcast a vector `b` (n,) against (n, 1) into an
    n x n "solution".
- Fix: one engine path. `solve(a, b)` resolves NumPy's result dtype (ints → float64,
  float16/32 → float32, complex kept), writes any structure/view/dtype out to dense, and runs
  LAPACK `?gesv` for float32/float64/complex64/complex128; a vector `b` is solved as a column
  over its storage and returned as a vector. The Python wrapper has no shortcut and no
  fallback; `solve_triangular` checks the triangular claim and calls `solve`.

### F-069 `lstsq` used the normal equations with Aᵀ
- **Status:** fixed. (AᵀA)⁻¹Aᵀb squares the condition number and is wrong for complex A
  (needs Aᴴ). Now `pinv(a) @ b` through the SVD: NumPy's minimum-norm answer.

### F-070 `slogdet` overflowed and failed for complex matrices
- **Status:** fixed. It computed `float(det)`: inf/0 for large n (log of an overflowed
  determinant), TypeError for complex. The device's `determinant` is replaced by `lu_pivots`
  (U's diagonal + permutation parity); `determinant` multiplies them, the new native
  `slogdet` sums their logs (NumPy's (sign, logabsdet), complex unit sign for complex input).

### F-071 A complex factor on a conjugated view was conjugated too: `A.conj() * 1j`
- **Status:** fixed. Views read conj(raw x factor); multiplying a conjugated view by c must
  store conj(c) in the factor. `A.conj() * 1j` gave conj(A * 1j) = -1j * conj(A), likewise
  `A.H * 1j`. Both `multiply_scalar(complex)` overrides fixed.

### F-072 Saving a window saved the whole matrix
- **Status:** fixed. `save(A[1:4, 1:4])` wrote A's full storage (the view record has no
  offsets); `load` returned a 3x3 view of the wrong elements. Windows are written out first.

### F-073 Python `matmul` converted structured operands to float64
- **Status:** fixed (dead code removed). The Python wrapper (used by `pc.matmul`,
  `matrix_power`, `lstsq`) turned symmetric/antisymmetric operands into dense float64
  (int32 symmetric @ x came back float64) and had float64-only diagonal/triangular
  conversions plus a NumPy/per-element generic fallback. The engine dispatches every
  structure; the wrapper now calls `a @ b` (array-likes are imported first, no fallback).

### F-074 `~C` of a causal matrix complemented only the stored triangle
- **Status:** fixed. `~` on a structured integer/bit matrix kept the structure (NumPy
  complements every element: the lower triangle and diagonal become True). Structured and
  viewed operands are written out before the bitwise kernel.

### F-075 Structured int16/int32 operands were multiplied as zeros
- **Status:** fixed (session 8). `TriangularMatrix(int32) @ DenseMatrix(int32)` (and
  symmetric/diagonal int16/int32 operands) took `cpu.matmul.mixed_int_fallback_i32`, chosen
  by dtype, which reads dense storage through `DenseMatrix` casts; a structured operand
  failed the cast and contributed zeros. The kernel now requires dense operands; others take
  the generic integer path (correct; slow, atlas ○).
- Root cause class (see the session 8 LOG): kernels selected by one property (dtype) that
  silently assume others (dense, plain). Planned cure: normalise at the boundary (PLAN).

### F-076 Save/load lost `has_diagonal`: a triangular matrix with a diagonal reloaded shifted
- **Status:** fixed (session 8, atlas rerun). Same class as F-064: the file metadata had no
  `has_diagonal`, and `_open_storage` built every typed triangle as strict, so the packed
  rows were read with the wrong offsets. Now saved in the metadata and passed to the load
  constructor (bit triangles are always strict).

### F-077 `UnitVector.conj()` / `.T` moved the 1 to index 0
- **Status:** fixed. Same class as F-064: the generic `VectorBase::clone` rebuilt through the
  factory, whose mapper constructor read the active index from the seed before the seed was
  set. `UnitVector` now has its own `clone()`.

### F-078 The diagonal of `A.T` for antisymmetric A was -0.0 (`2 / A.T` gave -inf)
- **Status:** fixed. `A.T` is stored as the factor -1 (A.T = -A), which negated the zero
  diagonal; NumPy gives +0.0 there. The antisymmetric accessors return +0.0 on the diagonal.

### F-079 `pinv` by the normal equations; `svd`/`svdvals` and complex SVD went through NumPy
- **Status:** fixed (session 8, atlas). `pinv` computed (AᴴA)⁻¹Aᴴ: the condition number
  squared, wrong for ill-conditioned input (atlas: bool windows, float32 antisymmetric), with
  a NumPy SVD fallback behind it. `svdvals` and `svd(full_matrices=True)` (NumPy's default)
  always ran NumPy; the native SVD was real-only.
- Fix: the native SVD takes any dtype/structure/view (normalised to a dense LAPACK dtype) and
  runs one templated `?gesdd` for float32/float64/complex64/complex128, reduced or full
  (NumPy shapes). `svd`, `svdvals` and `pinv` (V S⁺ Uᴴ with NumPy's cutoff) use it.

### F-080 Matmul: 25 dispatch branches and 9 per-element device fallbacks (Phase N.1)
- **Status:** fixed (session 8). Matmul was the clearest case of the Phase N root cause:
  `dispatch_matmul` chose among ~25 branches by `dynamic_cast` to one class pair at a time
  (float64 diagonal, float64 triangular, int32 triangular, int16/int32 mixes, bit x four
  dtypes, ...), and the device added a hand-written loop for each pair it had no kernel for
  (`float16_fallback`, `mixed_float_fallback_f32`, `cf16_fallback`, `mixed_int_fallback_i32`,
  `generic_fallback_{f32,int,f64}`, `tri_i32`). Every new dtype or structure needed a new
  branch; the loops read through the virtual accessors (atlas: 519 of 847 correct matmul cells
  generic). `diag_x_dense_f64` / `dense_x_diag_f64` indexed raw storage and ignored a
  transposed dense operand (unreachable only because the front never sent one). Four member
  `multiply` functions (dense, triangular, bit-dense, identity) were parallel entry points.
- Fix: kernels with declared contracts (`CpuSolver::matmul`): dense `T x T -> T` for every
  float, complex and integer T (complex now through `cgemm`/`zgemm`, it was a scalar triple
  loop); upper triangular `T x T -> T` for every real T (one template); bit-packed x dense
  float64/float32/int32/int16 (one template); bit x bit popcount. Anything else throws. One
  normaliser in `dispatch_matmul` resolves the dtype and applies five rules (identity,
  bit-packed, triangular, diagonal, dense); the dense rule converts each operand once to the
  kernel dtype (float16 computes in float32, complex float16 in complex64). Triangular and
  diagonal results now keep their structure for every dtype with a class (was float64/int32
  only). Net -930 lines.
- Atlas: matmul 557 of 847 correct cells fast (was 328), 290 generic (was 519). The 290 are
  operand conversions through the virtual accessor (float16 / complex float16, conjugated
  complex, symmetric, antisymmetric, diagonal, identity): fixed by a typed converter (N.2).
  An interim classifier rule that counted up to 4n² accessor calls as "normalisation" was
  reverted after the timing pass measured such a conversion (F-083).
- Semantics (documented in the code): structured rules skip structural zeros, so they differ
  from NumPy only where a skipped 0·inf or 0·nan would give nan. Integer matmul raises on
  overflow (F-055 policy) where NumPy wraps: unchanged, listed for the maintainer.
- Tests: `tests/engine/test_matmul_rules.py` (result structure per rule, overflow guards).

### F-081 Cached results went stale on mutation; trace/det answered from user flags
- **Status:** fixed (session 8, found by the timing pass). The Python properties layer cached
  `sum`, `trace`, `determinant`, `norm`, `rank`, eigenvalues/eigenvectors and the inverse on the
  object, propagated them through `*`, `+`, `-`, `.T`, `.conj()` without computing, and persisted
  them in save files (plus "big-blob" sibling files for inverses and eigenvectors). Invalidation
  hooked `.set()` only, so after `M[0, 0] = 100` `pc.sum(M)` and `pc.determinant(M)` returned
  the old values (66 instead of 156). The `trace()` / `determinant()` wrappers also answered
  from user-set flags (`is_zero` → det 0, `is_identity` → trace n, `diagonal_value` →
  det dv**n) without computing: properties changing results, against the approved rule.
  `m.invert()` on a file-backed float64 matrix was forced through NumPy (a stale comment
  claimed the native inverse was wrong there; the atlas file-storage cells show it is right),
  and any native inverse error fell back to NumPy silently.
- Fix: all derived-value caching removed (properties.py cached keys, propagation and the
  trace/determinant wrappers; `linalg_cache.py`, `big_blob_cache.py`; the `cached` metadata
  namespace; eigen cache reads/writes in ops.py; the norm/sum caches in `__init__`). Results
  are computed on every call. User properties remain as metadata (their removal is N.3).
  Old tests that asserted caching (including two named `..._prefers_properties_even_if_wrong`)
  deleted; docs (storage guide, container format, API contract, Adding Operations protocol)
  rewritten. Regression: `test_F081_*` (results follow `M[i,j] = v` and `.set`; flags never
  change trace/det/sum).

### F-082 `pc.symmetric` / `pc.antisymmetric` / `pc.diagonal` predate the structure classes
- **Status:** open. They return a dense matrix with a property flag for integer input (not a
  `SymmetricMatrix`), fill floats with per-element Python loops, and their docstrings
  describe the pre-F-032 classes ("SymmetricMatrix is float64"). To be rebuilt on the
  structure classes with a native fill, or removed (the classes' constructors cover them).

### F-083 "fast" was a path label, not a speed; per-call overhead 30-50 µs
- **Status:** measured (session 8, timing pass `tests/atlas/timing.py`, `review/atlas/TIMING.md`).
  The atlas's `fast` only meant "no per-element virtual calls". Measured against NumPy on the
  same data: see TIMING.md for the per-operation medians. Causes found so far: (a) per-call
  overhead of the Python/binding layers, 30-50 µs per `+` or `@` vs NumPy's 0.3-0.6 µs; (b)
  operand conversion through the virtual accessor (~100 ns/element; float32 antisymmetric @ X
  at n = 384: 17.5 ms vs 0.28 ms); (c) `bool @ bool.T` took the per-element branch of the
  popcount kernel (104 ms vs 1.4 ms).
- Policy (ATLAS.md §5): an `ok` cell must reach ≥ 0.9× NumPy at the reference size, or carry a
  finding that explains why.

