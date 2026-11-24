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
- **Status:** open
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
- **Fix direction:** decide one rule. Either (a) all readers honour `scalar_`,
  `is_conjugated_` and `is_transposed_` through one accessor, or (b) integer types
  materialize on scale. Then audit every `get_element*`, export and kernel entry for
  view metadata. Add a property test: for every dtype and op, element access ==
  export == NumPy.

### F-004 Nested `ParallelFor` deadlocks
- **Status:** open
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
- **Status:** open
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
- **Status:** open
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
- **Status:** open
- `pc.cond([[4,1],[2,3]])` = 3.0, `np.linalg.cond` = 2.618. Any `p=` raises
  NotImplementedError. This violates the "speaks NumPy" contract silently.
- **Where:** `python/pycauset/_internal/ops.py:852`.

### F-005 `ParallelFor` exception path can use-after-free
- **Status:** open (found by reading the code; not reproduced)
- When one chunk throws, `fut.get()` rethrows in the caller while other workers keep
  running `func`, which usually captures the caller's stack by reference. The caller
  unwinds and those references dangle. Fix: wait for all futures, then rethrow the
  first exception.
- Also: `ThreadPool::global_num_threads` is a non-atomic static read across threads.

### F-006 Signed-integer-overflow UB in int32 elementwise add
- **Status:** open
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
- **Status:** open
- `tests/test_cpu_worker_contract.cpp`, `tests/test_op_registry.cpp` and
  `tests/test_parallel_utils.cpp` are not in CMakeLists. The first two pass when
  built by hand; the third reveals F-004. CI builds **no** C++ tests at all.

## S3 — minor

### F-011 Several "native" linalg ops always route through `numpy.linalg`
- **Status:** open
- `eigh`, `eigvalsh`, `svdvals` and `matrix_rank` call `numpy.linalg` (full in-RAM
  copy) even though `CpuSolver` implements `eigh`/`eigvalsh`/`svd`. They cannot run
  out-of-core. See `review/probes/numpy_fallback_probe.py`.

### F-012 Elementwise and vector ops are 2–5× slower than NumPy
- **Status:** open
- `benchmarks/r2_parity.py` on this machine (4 vCPU): add 0.22×, multiply 0.29×,
  dot 0.47×. matmul, solve and eigh are at parity; invert and det are 3× faster.
  The project's own bar is ≥ 0.90×. Suspects: the Python wrapper stack (§4.2),
  allocating a fresh result plus temp-file registration per op, and thread-pool
  startup per call.

### F-013 bool dtype semantics diverge from NumPy without documentation
- **Status:** open (it may be intentional for causal matrices, but it must be documented)
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
- **Status:** open (UX decision)
- `np.asarray(M)` raises for any file-backed object, regardless of size. A 300×300
  matrix spilled by the governor can't be printed or converted without
  `pc.to_numpy(M, allow_huge=True)`. The spill decision is automatic, so the user
  can't predict when this will happen.

### F-017 Default spill directory is `./.pycauset` in the current working directory
- **Status:** open (UX decision)
- Running a script from a repo or home directory creates spill files there.

## S4 — hygiene

### F-018 Dead code
- **Status:** open
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
- **Status:** open
- `#pragma message("Spacetime.hpp included")` prints on every compile.
- `_GNU_SOURCE` is redefined in `src/core/MemoryMapper.cpp:1`.

### F-020 Compiler warnings under `-Wall -Wextra`
- **Status:** open
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
- **Status:** open (S4: UB, harmless in practice)
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
- **Status:** open (S3: CI hygiene)
- Non-fatal UBSan reports go to fd 2, which pytest captures. Without `-s` a
  sanitized run looks clean even when UB occurs (verified 2026-11-24). `asan.yml`
  passes `-s`, but it runs ASan only (no UBSan) and only on `main`. The new CI
  sanitizer job must use `-s` or `-fno-sanitize-recover=undefined`.
