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
0.3.34), `benchmarks/r2_parity.py`, float64, N = 1024, 2026-11-27:

| op | ratio |
|---|---|
| `A @ B` | 0.94x |
| `A + B`, `A * B` | 2.8x |
| `dot`, export to NumPy | 1.0x |
| `solve` | 0.50-0.56x |
| `invert` | 0.61-0.66x |
| `determinant` | 0.65-0.69x |
| `eigh` | 0.52-0.63x |

The LAPACK operations lose half their time to row-major LAPACKE calls, which transpose
every operand in and out; called column-major on the same buffer the factorization runs 2x
faster than NumPy's (review FINDINGS F-108, PLAN 6.0a). Most other cells of the atlas
timing are below parity too, small operations most of all (per-call overhead).

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
