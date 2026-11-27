# PyCauset Benchmarks

PyCauset against NumPy, measured. Speed work is review Phase 6; until it lands, these
numbers are the honest state, not a goal.

## How to measure

| Command | What it measures |
|---|---|
| `python benchmarks/r2_parity.py` | headline dense float64 ops at N = 1024 (CI runs it, report-only) |
| `python -m tests.atlas.timing [op ...]` | every op x dtype x structure x view against NumPy; writes `review/atlas/TIMING.md` |
| `python benchmarks/bench.py [--large]` + `python benchmarks/plot.py` | time against size, with graphs |
| `python benchmarks/benchmark_numpy_parity.py` | NumPy import / export throughput |
| `python benchmarks/benchmark_io_smoke.py` | save / load wall time |
| `python benchmarks/bench_ram.py` | a matrix larger than the RAM budget, file-backed |
| `python benchmarks/scenario_benchmarks.py`, `large_n_check.py` | causal-set workloads |

Ratios are `numpy_time / pycauset_time`: above 1 means PyCauset is faster. Compare runs on
the same machine only; BLAS timings differ a lot between machines and OpenBLAS builds.

## Current numbers

4-vCPU Linux container, system OpenBLAS 0.3.26 (NumPy 2.x with its bundled OpenBLAS
0.3.34), float64, N = 1024 (n = 512 for `eig`), best of 4, 2026-11-27, PyCauset imported
first. `benchmarks/r2_parity.py`: 9/9 operations at or above 0.90x.

| op | NumPy | PyCauset | ratio |
|---|---|---|---|
| `A @ B` | 7.5 ms | 8.1 ms | 0.90-0.93x |
| `A + B`, `A * B` | 1.3 ms | 0.45 ms | 2.7-3.2x |
| `dot`, export to NumPy | | | 0.95-1.0x |
| `solve` (matrix right-hand side) | 16 ms | 9.4 ms | 1.7x |
| `solve` (vector) | 17 ms | 11 ms | 1.6x |
| `invert` | 40 ms | 25 ms | 1.6x |
| `determinant` / `slogdet` | 15-17 ms | 8.5-9.6 ms | 1.8x |
| `cholesky` | 26 ms | 11 ms | 2.4x |
| `eigh` / `eigvalsh` | 110 / 49 ms | 115 / 47 ms | 0.96-1.05x |
| `qr` | 126 ms | 123 ms | 1.0x |
| `svd` / values only | 436 / 219 ms | 343 / 166 ms | 1.3x |
| `pinv`, `matrix_rank` | 425 / 181 ms | 354 / 145 ms | 1.2x |
| `eig` (n = 512) | 207 ms | 211 ms | 1.0x |
| `lstsq` (vector) | 253 ms | 210 ms | 1.2x |

Two fixes got the LAPACK operations here from 0.5-0.7x (review F-108): LAPACK is called
column-major on the row-major buffers (LAPACKE's row-major interface transposed every
operand in and out), and OpenBLAS threads spin for about 0.1 ms after a call instead of
going to sleep at once (`OPENBLAS_THREAD_TIMEOUT=18`; factorizations are many small BLAS
calls, and waking the threads for each one cost 2x). Small operations elsewhere in the
atlas timing are still below parity (per-call overhead).

## Where PyCauset is ahead by construction

- **Causal (bit) matrices:** one bit per relation, 8x smaller than a NumPy `bool` array and
  64x smaller than float64; `C @ C` counts paths with word-level popcount.
- **Structure:** identity and diagonal products, inverses and determinants are O(N) or
  O(1); triangular, symmetric and antisymmetric matrices store about half the elements; a
  lazy `C + C.T` is never written out.
- **Past RAM:** NumPy holds every array in RAM (a float64 matrix on a 16 GB machine stops
  near 46,000 x 46,000). PyCauset memory-maps matrices above its threshold to disk and
  streams matmul and element-wise operations tile by tile (`bench_ram.py`: a 1.07 GB matrix
  with a 256 MB budget).
