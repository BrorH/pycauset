# R2_PERF — parity findings (R2.2)

Measured with `python benchmarks/r2_parity.py` (dense float64, n=1024 matrices,
best-of-3, PyCauset result forced through `np.asarray`; `invert`/`determinant`
clear their derived caches before timing so both sides recompute). The ratio is
`numpy_time / pycauset_time`; the R2_PERF bar is **≥ 0.90×**.

Current state (threaded DYNAMIC_ARCH OpenBLAS, `OPENBLAS_NUM_THREADS=8` default,
GEMM bumps to 20 threads internally):

| op | ratio | verdict |
| :-- | --: | :-- |
| add | ~1.1–1.2× | PASS |
| solve | ~0.96× | PASS |
| invert | ~1.1× | PASS |
| dot | ~2× | PASS |
| multiply | ~0.93× | PASS (marginal) |
| determinant | ~1.24× | PASS |
| eigh | ~0.98× | PASS |
| matmul | ~0.87× | FAIL (marginal) |

7/8 ops are at or above parity. `matmul` hovers just under 0.90× (~0.0071 s vs
NumPy ~0.0063 s) and `multiply` just above it (~0.93×); both flap across the 0.90
line run-to-run (CPU frequency/thermal noise).

## What landed in R2.2

1. **`determinant` → LAPACK** — `CpuSolver::determinant` uses `LAPACKE_dgetrf`/
   `zgetrf` (NumPy's own backend) instead of Eigen's single-threaded `PartialPivLU`.
2. **Fair `invert`/`determinant` benchmarking** — the benchmark clears the derived
   caches (`_invalidate_cached_derived`) before timing.
3. **`add` flakiness root-caused** — a stale installed wheel merged into a namespace
   package and `import_native_extension()` used `next(iter(pkg.__path__))` (filesystem-
   dependent order on Windows), so ~50% of processes loaded a stale `_pycauset.pyd`.
   Fixed by loading from the package's `__file__` directory (`native.py`).
4. **AVX2 f64 sub/mul/div kernels + full-span hardening** — elementwise `subtract`/
   `multiply`/`divide` now route through AVX2 with strict full-span guards.
5. **`matmul` OpenBLAS rebuilt** — the CMake fallback downloaded a *generic single-core*
   0.3.26 binary; rebuilt 0.3.28 with `DYNAMIC_ARCH=1 USE_THREAD=1` (Haswell kernels),
   relinked, and bundled the MinGW runtime DLLs. `build_openblas.ps1` documents it.
6. **`OPENBLAS_NUM_THREADS` default 8** — the threaded OpenBLAS uses one pool for BLAS
   and LAPACK; 20 threads regresses small LAPACK (invert/determinant) via SMP sync
   overhead, so `pycauset` pins 8 at import and the matmul GEMM temporarily bumps to 20.

## The two memory/perf root fixes (biggest wins)

- **`:memory:` backing switched to `VirtualAlloc`** (`MemoryMapper`) instead of a
  pagefile-backed section (`CreateFileMappingA(INVALID_HANDLE_VALUE)`). The pagefile
  path commits a page from the paging file on every first write, costing several ms
  per 8MB result and dominating small elementwise ops. This single change took
  `add` 0.88× → ~1.1× and `multiply` 0.72× → ~0.93×.
- **CPU dgemm no longer pins operands** (`CpuSolver::attempt_direct_path`) — `VirtualLock`
  is a GPU-DMA optimization and pure overhead for CPU `dgemm`.

## Remaining gap: `matmul` (~0.87×)

The raw `cblas_dgemm` is already **faster than NumPy** (0.0044 s vs NumPy's 0.0060 s
at the same 20 threads). The residual is per-call wrapper cost on top of the kernel:

| component | time |
| :-- | --: |
| raw `cblas_dgemm` (warm buffer, 20 threads) | ~0.0044 s |
| fresh 8MB result (VirtualAlloc + first-write faults) | ~1.6 ms |
| C++ `create_matrix` + MemoryGovernor + dispatch | ~0.4 ms |
| Python `_ops.matmul` (IO planning, streaming checks, inline imports) | ~0.7 ms |
| **total** | **~0.0071 s** |

NumPy's `@` skips most of the Python/IO-planning layer. Fully closing the 0.03× gap
means trimming the Python `_ops.matmul` IO-planning/inline-import path (or routing the
native `matmul` through the eager operator), which trades away the streaming/observability
layer for the hot dense case. Tracked against `R2_PERF` in `R2_ROADMAP.md`.
