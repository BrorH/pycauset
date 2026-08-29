# R2_PERF — parity findings (R2.2)

Measured with `python benchmarks/r2_parity.py` (dense float64, n=1024 matrices,
best-of-3, PyCauset result forced through `np.asarray`; `invert`/`determinant`
clear their derived caches before timing so both sides recompute). The ratio is
`numpy_time / pycauset_time`; the R2_PERF bar is **≥ 0.90×**.

Current state (threaded DYNAMIC_ARCH OpenBLAS, `OPENBLAS_NUM_THREADS=8` default):

| op | ratio | verdict |
| :-- | --: | :-- |
| solve | 0.96× | PASS |
| invert | 1.09× | PASS |
| dot | ~1.8× | PASS |
| determinant | 1.24× | PASS |
| eigh | 0.98× | PASS |
| add | 0.88× | FAIL (marginal) |
| multiply | 0.72× | FAIL |
| matmul | 0.80× | FAIL |

## What landed in R2.2

1. **`determinant` → LAPACK** — `CpuSolver::determinant` now uses `LAPACKE_dgetrf`/
   `zgetrf` (NumPy's own backend) instead of Eigen's single-threaded `PartialPivLU`.
   This took determinant from ~3× slower to ~1.24× (PASS).
2. **Fair `invert`/`determinant` benchmarking** — the benchmark clears the derived
   caches (`_invalidate_cached_derived`) before timing, so a warm inverse/determinant
   cache is no longer compared against NumPy's fresh factorization.
3. **`add` flakiness root-caused** — NOT a C++ Heisenbug. A stale installed wheel in
   `site-packages` (no `__init__.py`) merged with the source checkout into a namespace
   package, and `import_native_extension()` used `next(iter(pkg.__path__))`, whose
   entry order is filesystem-dependent on Windows. ~50% of processes loaded a stale
   `_pycauset.pyd` (predating the lazy `try_eval_fast` SIMD path) and took the blocked
   `target = expr_` eval (~0.019 s) instead of the AVX2 path (~0.003 s). Fixed by
   loading from the package's own `__file__` directory (see `python/pycauset/_internal/native.py`).
4. **AVX2 f64 sub/mul/div kernels + full-span hardening** — `try_fast_simd` now routes
   `subtract`/`elementwise_multiply`/`elementwise_divide` through AVX2 (was only `add`),
   with stricter full-span guards (zero-offset strided views fall back correctly).

## `matmul` — bundled OpenBLAS build/tuning (0.56× → 0.80×)

`pc.matmul` has always hit `cblas_dgemm` (not a naive loop). The gap was the bundled
OpenBLAS: the CMake fallback downloads `OpenBLAS-0.3.26-x64.zip`, a **generic**
(single-core, no arch-specific kernels) build. NumPy links `scipy-openblas` (DYNAMIC_ARCH,
Haswell, threaded).

Fix attempted in R2.2: build OpenBLAS 0.3.28 from source with
`DYNAMIC_ARCH=1 USE_THREAD=1 NUM_THREADS=24` (MinGW-w64 gcc/gfortran 15.2, see
`build_openblas.ps1`), generating an MSVC import lib via `gendef`+`dlltool`, and relinking
`pycauset_core.dll`. The MinGW runtime DLLs (`libgcc_s_seh-1`, `libgfortran-5`,
`libquadmath-0`, `libwinpthread-1`) must be bundled alongside `libopenblas.dll`.

Result: `matmul` 0.66× → ~0.80× (0.011 s → ~0.008 s at n=1024), `corename=Haswell`.

### OpenBLAS thread-count tradeoff (important)

The threaded OpenBLAS uses one thread pool for BLAS **and** LAPACK. At the default
(20 threads on this machine) small LAPACK factorizations (`dgetrf`/`dgetri` for
`invert`/`determinant`) pay SMP-server sync overhead and become **slower** than
single-threaded. `pycauset` therefore pins `openblas_set_num_threads(8)` at import
(overridable via `OPENBLAS_NUM_THREADS`), which keeps `invert`/`determinant` at parity
while still speeding GEMM. At n=1024, 8 threads ≈ 0.80× matmul; more threads help GEMM
only marginally but hurt LAPACK.

Remaining `matmul` gap to 0.90× is the OpenBLAS version/tuning delta (0.3.28 vs NumPy's
0.3.30 `scipy-openblas`); fully closing it needs either NumPy's exact OpenBLAS (its DLL is
`scipy_`-symbol-prefixed, so a relink through a symbol shim) or a 0.3.30+ tuned build.

## Remaining below-parity ops

- **`add` (0.88×)** — essentially at parity; the residual is the `:memory:` backing
  (pagefile-backed `CreateFileMappingA(INVALID_HANDLE_VALUE)` + `MemoryGovernor`
  `request_ram`/`register_object` on every result allocation) vs NumPy's `malloc`.
- **`multiply` (0.72×)** — eager `elementwise_multiply` (the MatrixBase `operator*` for
  matrix×matrix is eager, unlike the lazy `+`/`-`/`÷`). Same SIMD kernel as `add`, but the
  eager free-function path (~0.0030 s) is ~15% slower than the lazy add path (~0.0026 s).
- **`matmul` (0.80×)** — OpenBLAS tuning (see above).

These are tracked against the R2E engine track in `R2_ROADMAP.md` (`R2_PERF`).
