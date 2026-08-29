# R2_PERF, parity findings (R2.2)

Measured with `python benchmarks/r2_parity.py` (dense float64, n=1024 matrices,
best-of-3, PyCauset result forced through `np.asarray`; `invert`/`determinant`
clear their derived caches before timing so both sides recompute). The ratio is
`numpy_time / pycauset_time`; the R2_PERF bar is **≥ 0.90×**.

**State: 8/8 ops at parity.** Stable across repeated runs:

| op | ratio | verdict |
| :-- | --: | :-- |
| add | ~1.1–1.25× | PASS |
| solve | ~0.97–0.99× | PASS |
| invert | ~1.1–1.2× | PASS |
| dot | ~1.4–4× | PASS |
| multiply | ~1.1–1.2× | PASS |
| determinant | ~1.2–1.3× | PASS |
| eigh | ~0.98–0.99× | PASS |
| matmul | ~0.94–0.96× | PASS |

## What landed in R2.2 (parity work)

1. **`determinant` → LAPACK**, `LAPACKE_dgetrf`/`zgetrf` instead of Eigen's
   single-threaded `PartialPivLU`.
2. **Fair `invert`/`determinant` benchmarking**, benchmark clears derived caches.
3. **`add` flakiness root-caused**, stale-binary namespace merge (`native.py` now
   loads from `__file__`).
4. **AVX2 f64 sub/mul/div kernels + full-span hardening**.
5. **OpenBLAS rebuilt**, 0.3.28 DYNAMIC_ARCH+threaded (Haswell) replacing the
   generic single-core 0.3.26; MinGW runtime DLLs bundled (`build_openblas.ps1`).
6. **`:memory:` backing → `VirtualAlloc`**, the pagefile-backed section committed a
   page from the paging file on every first write (the add/multiply win).
7. **CPU `dgemm` no longer pins operands**, `VirtualLock` is GPU-only overhead.
8. **GEMM bumps to 20 threads**, dgemm scales with threads, LAPACK doesn't.
9. **`mark_temporary_if_auto` skips `Path.resolve()` for `:memory:`**, the
   `nt._getfinalpathname` syscall (~30–60 µs) ran on every matrix construction and
   was the final matmul residual (0.87× → 0.94×).

## The decisive fixes (biggest wins)

- **`:memory:` `VirtualAlloc`** (add 0.88→1.1×, multiply 0.72→~1.1×).
- **`Path.resolve()` short-circuit** (construction 0.38 ms → 0.007 ms; matmul
  0.87→0.94×, and every other op improved too, the temp-file tracking was the
  single biggest per-call overhead across the whole op surface).
- **Threaded DYNAMIC_ARCH OpenBLAS** (matmul 0.56→0.94× overall).

The raw `cblas_dgemm` is faster than NumPy (0.0044 s vs 0.0060 s at 20 threads);
with the wrapper overhead removed, `pc.matmul` now lands at ~0.94×.

