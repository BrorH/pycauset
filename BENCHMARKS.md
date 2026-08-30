# PyCauset Benchmarks

Measured performance of PyCauset against NumPy, for every operation that has a
direct NumPy equivalent. These numbers are reproducible on your machine:

```
python benchmarks/bench.py --large
python benchmarks/plot.py
```

## Methodology

- **Hardware:** Intel Core i9-10850K (10 cores @ 3.6 GHz), 32 GB RAM, Windows 11.
- **Versions:** NumPy 2.3.5 (OpenBLAS), PyCauset 0.6.2.dev0 (OpenBLAS 0.3.28).
- **Timing:** `time.perf_counter`, best-of-N per operation (fewer repeats at large n).
- **Dtype:** dense float64.
- **Ratio:** `numpy_time / pycauset_time`. Values above 1.0x mean PyCauset is faster.

## R2.2 parity gate: 8/8 ops at or above 0.90x NumPy

The R2.2 gate is `benchmarks/r2_parity.py`: every operation with a direct NumPy
equivalent must run at least 0.90x as fast as NumPy. `invert` and `determinant`
clear their derived caches on every timed call so the comparison is a fresh
factorization on both sides, not a cache hit. Median ratio over 6 runs (n=1024):

| op | x times faster than numpy |
|---|---|
| matmul | 1.03x |
| solve | 1.02x |
| invert | 1.19x |
| determinant | 1.26x |
| add | 1.31x |
| multiply | 1.22x |
| dot | 2.45x (noisy, ~0.1 ms op) |
| eigh | 0.99x |

```text
  matmul       1.03x  ██████████
  solve        1.02x  ██████████
  invert       1.19x  ███████████▉
  determinant  1.26x  ████████████▋
  add          1.31x  █████████████▏
  multiply     1.22x  ████████████▏
  dot          2.45x  ████████████████████████▌   (noisy)
  eigh         0.99x  █████████▉
  -------- parity 1.00x  ██████████
```

Where the boost came from, versus the pre-R2.2 baseline in the tables below:

- **add / multiply / elementwise** moved from ~0.54x / ~0.72x to ~1.3x / ~1.2x by
  running the AVX2 SIMD kernels over fast anonymous `VirtualAlloc` backing (instead
  of a pagefile-backed section) and by keeping CPU matrices on ordinary pageable
  memory when the GPU is active (pinning is on-demand now).
- **matmul / solve / invert / determinant** reached parity by rebuilding OpenBLAS
  0.3.28 with `DYNAMIC_ARCH` + threading and linking it by name (the CUDA build had
  been linking a stale ordinal-based import lib), and by clearing the invert/determinant
  derived caches so the gate measures a real factorization.
- **eigh** was already at parity; the gate now also holds with the GPU backend active.

## Dtype coverage (correctness)

`benchmarks/bench_coverage.py` checks every operation against NumPy across all
supported dtypes and classifies each cell as `ok`, `by-design` (documented error), or
`WRONG`. Current status: every operation passes all its supported dtypes.

| op | dtypes | result |
|---|---|---|
| matmul, add, norm, trace | 14 (bit + all int/uint + all float/complex) | all ok |
| invert, solve, cholesky, svdvals, matrix_rank, determinant, eigvalsh | 4 (float32/64, complex64/128) | all ok |
| matrix_power, outer | 6 (float/complex + int32/64) | all ok |

## Structural shortcuts (instant metadata calculations)

PyCauset stores semantic metadata on matrices (properties-as-gospel). When a matrix is
marked as identity, diagonal, triangular, or zero, operations that would otherwise be
O(n^3) reduce to O(1) or O(n) closed forms. Measured at n=800 (shortcut vs the general
path on identical data, via `benchmarks/bench_structures.py`):

| operation on a structured matrix | shortcut vs general | NumPy does |
|---|---|---|
| `matrix_rank(identity)` | 700x faster | full SVD |
| `matrix_rank(diagonal)` | 126x faster | full SVD |
| `matrix_rank(triangular)` | 157x faster | full SVD |
| `norm(identity)` | 314x faster | full Frobenius pass |
| `norm(diagonal)` | 734x faster | full Frobenius pass |
| `determinant(identity/diagonal/triangular)` | 200-240x faster | LU |
| `matrix_power(identity/zero)` | 200-240x faster | repeated matmul |
| `invert(identity)` | 24x faster | LU |

This is where PyCauset wins even though its dense kernels run at NumPy parity: a large
identity or diagonal matrix gets an answer instantly instead of spending seconds on an
SVD or LU factorization, and NumPy has no equivalent metadata shortcut at all.

## Symmetric / anti-symmetric storage (2x smaller)

`pycauset.symmetric(data)` and `pycauset.antisymmetric(data)` construct native
`SymmetricMatrix` / `AntiSymmetricMatrix` objects that store only the upper triangle
(including the diagonal), so an n x n float64 matrix costs roughly n(n+1)/2 elements
instead of n^2. For large n this is about a 2x reduction in RAM and disk footprint,
independent of the out-of-core threshold, and `to_numpy`/`save`/`load` reconstruct the
full matrix on demand. Optimization of symmetric/antisymmetric kernels is part of the
post-R1 program.

## Summary

PyCauset's dense kernels run on the same OpenBLAS/LAPACK backend as NumPy, so the
realistic goal is parity, not a large speedup. As of R2.2 every parity-gate operation
runs at or above 0.90x NumPy (see the table above), and the elementwise ops that used to
lag (add, multiply) now edge ahead. The remaining gaps are SVD (about 2x, from the
row-major transpose) and out-of-core factorizations, both tracked in `TODO.md`.

The detailed tables and graphs below are the pre-R2.2 baseline, kept for their per-size
detail; the current numbers are the R2.2 parity table and bar chart above.

## The reason to use PyCauset: it scales past RAM

NumPy keeps every array in RAM, so a square float64 matrix is capped by memory: on a
16 GB machine that is about 46340 x 46340 (16.2 GB), and beyond that NumPy raises
`MemoryError`. PyCauset memory-maps to disk beyond a RAM budget, so its RAM usage is
bounded by your threshold and the matrix can be far larger than RAM.

| RAM budget | max n (NumPy) | PyCauset |
|---|---|---|
| 4 GB | 23170 x 23170 | unbounded (disk-backed) |
| 8 GB | 32768 x 32768 | unbounded (disk-backed) |
| 16 GB | 46340 x 46340 | unbounded (disk-backed) |
| 32 GB | 65536 x 65536 | unbounded (disk-backed) |
| 64 GB | 92681 x 92681 | unbounded (disk-backed) |

Demonstration (run `python benchmarks/bench_ram.py`): a 12000 x 12000 float64 matrix is
1.07 GB; NumPy holds all of it in RAM, while PyCauset with a 256 MB budget stores it on
disk and still computes `trace(identity) = 12000` correctly. This is the regime where
PyCauset does something NumPy cannot, which is the practical reason to use it even where
NumPy is a bit faster on in-memory matrices.

## Time vs n

![matmul, inverse, solve](documentation/docs/assets/benchmarks/time_matmul_fact.png)

![cholesky, eigh, svd](documentation/docs/assets/benchmarks/time_eigen_svd.png)

![elementwise add, dot](documentation/docs/assets/benchmarks/time_elem_dot.png)

## Speedup by operation

![speedup by operation](documentation/docs/assets/benchmarks/speedup_by_op.png)

## Dense float64 matmul (C = A @ B)

| n | NumPy | PyCauset | speedup |
|---|---|---|---|
| 256 | 1.9ms | 4.0ms | 0.46x |
| 512 | 7.2ms | 10.4ms | 0.69x |
| 1024 | 32.5ms | 42.6ms | 0.76x |
| 2048 | 173.7ms | 188.9ms | 0.92x |
| 4096 | 968.9ms | 990.4ms | 0.98x |
| 8192 | 6.11s | 5.92s | **1.03x** |

Matmul crosses parity around n = 4096 and edges ahead at n = 8192. Small sizes carry
fixed Python dispatch overhead.

## Dense float64 factorizations

| op | n | NumPy | PyCauset | speedup |
|---|---|---|---|---|
| inverse | 256 | 4.2ms | 4.9ms | 0.86x |
| inverse | 512 | 12.4ms | 13.6ms | 0.92x |
| inverse | 1024 | 66.8ms | 94.4ms | 0.71x |
| inverse | 2048 | 348.3ms | 371.7ms | 0.94x |
| inverse | 4096 | 1.71s | 2.22s | 0.77x |
| solve | 256 | 1.4ms | 2.6ms | 0.54x |
| solve | 512 | 6.0ms | 9.2ms | 0.65x |
| solve | 1024 | 31.5ms | 40.2ms | 0.78x |
| solve | 2048 | 190.8ms | 261.8ms | 0.73x |
| solve | 4096 | 1.19s | 1.46s | 0.81x |
| cholesky | 256 | 2.0ms | 2.8ms | 0.71x |
| cholesky | 512 | 7.3ms | 9.4ms | 0.77x |
| cholesky | 1024 | 35.4ms | 46.8ms | 0.75x |
| cholesky | 2048 | 173.5ms | 165.9ms | **1.05x** |
| cholesky | 4096 | 959.5ms | 1.01s | 0.95x |
| svd | 256 | 19.2ms | 25.7ms | 0.75x |
| svd | 512 | 67.2ms | 193.3ms | 0.35x |
| svd | 1024 | 315.6ms | 541.3ms | 0.58x |
| svd | 2048 | 2.42s | 4.74s | 0.51x |

Inverse, solve, and Cholesky are competitive (0.7x to 1.05x). SVD now uses the same
divide-and-conquer LAPACK routine (`gesdd`) as NumPy; the remaining roughly 2x gap is the
row-major transpose overhead and is tracked for the post-R1 program.

## Dense float64 eigenvalues

| op | n | NumPy | PyCauset | speedup |
|---|---|---|---|---|
| eigh | 256 | 11.0ms | 13.0ms | 0.84x |
| eigh | 512 | 46.7ms | 49.1ms | 0.95x |
| eigh | 1024 | 157.8ms | 161.6ms | 0.98x |
| eigh | 2048 | 840.4ms | 831.1ms | **1.01x** |
| eigvalsh | 256 | 6.5ms | 7.6ms | 0.86x |
| eigvalsh | 512 | 32.2ms | 33.8ms | 0.95x |
| eigvalsh | 1024 | 107.9ms | 109.0ms | 0.99x |
| eigvalsh | 2048 | 447.7ms | 465.4ms | 0.96x |

Eigenvalue routines reach parity (0.95x to 1.01x) across the measured range.

## Elementwise add and dot

| op | n | NumPy | PyCauset | speedup |
|---|---|---|---|---|
| add | 1024 | 28.6ms | 51.3ms | 0.56x |
| add | 2048 | 107.1ms | 193.0ms | 0.55x |
| add | 4096 | 440.6ms | 811.4ms | 0.54x |
| add | 8192 | 1.91s | 3.18s | 0.60x |
| dot | 100000 | 0.087ms | 0.014ms | **6.21x** |
| dot | 1000000 | 0.168ms | 0.210ms | 0.80x |
| dot | 10000000 | 5.72ms | 6.05ms | 0.95x |

Dot is at parity or faster (vectors are pre-constructed; the earlier gap was vector
construction overhead, not the dot itself). Elementwise add remains below parity because
the lazy-expression materialization path is not yet SIMD-optimized; it is a post-R1 target.

## What this means

- **Every parity-gate operation is now at or above 0.90x NumPy** (R2.2, see the table at
  the top): matmul, solve, invert, determinant, add, multiply, dot, and eigh all pass.
- **SVD** remains the one dense gap, roughly 2x (down from 25x after switching to
  `gesdd`), from the row-major transpose overhead; tracked in `TODO.md`.
- **Where PyCauset wins outright** is the structural/metadata shortcuts (hundreds of x
  on identity/diagonal/triangular matrices) and scaling past RAM, which NumPy cannot do.
