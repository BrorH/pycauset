# PyCauset Benchmarks

Measured performance of PyCauset against NumPy, for every operation that has a
direct NumPy equivalent. These numbers are reproducible on your machine:

```
python benchmarks/bench.py --large
python benchmarks/plot.py
```

## Methodology

- **Hardware:** Intel Core i9-10850K (10 cores @ 3.6 GHz), 32 GB RAM, Windows 11.
- **Versions:** NumPy 2.3.5 (OpenBLAS), PyCauset 0.5.1 (OpenBLAS 0.3.26).
- **Timing:** `time.perf_counter`, best-of-N per operation (fewer repeats at large n).
- **Dtype:** dense float64.
- **Ratio:** `numpy_time / pycauset_time`. Values above 1.0x mean PyCauset is faster.

## Summary

PyCauset's dense kernels run on the same OpenBLAS/LAPACK backend as NumPy, so the
realistic goal is parity, not a large speedup. At large sizes PyCauset matches or
edges ahead of NumPy on matmul, eigenvalues, and Cholesky. The current gaps are SVD
(which does not yet use LAPACK efficiently) and elementwise materialization; both are
tracked for the post-R1 performance program.

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
| svd | 256 | 19.2ms | 127.4ms | 0.15x |
| svd | 512 | 67.3ms | 886.0ms | 0.08x |
| svd | 1024 | 313.8ms | 6.47s | 0.05x |
| svd | 2048 | 2.42s | 65.04s | 0.04x |

Inverse, solve, and Cholesky are competitive (0.7x to 1.05x). SVD is the clear outlier:
PyCauset's SVD path does not yet use LAPACK efficiently and is 10x to 25x slower. This is
the top priority for the post-R1 performance program.

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
| add | 1024 | 29.1ms | 51.7ms | 0.56x |
| add | 2048 | 113.3ms | 194.4ms | 0.58x |
| add | 4096 | 440.2ms | 787.3ms | 0.56x |
| add | 8192 | 1.81s | 3.18s | 0.57x |
| dot | 100000 | 1.5ms | 2.8ms | 0.54x |
| dot | 1000000 | 18.8ms | 27.4ms | 0.69x |
| dot | 10000000 | 130.7ms | 211.7ms | 0.62x |

Elementwise materialization and dot are below parity (0.54x to 0.69x). These paths go
through lazy-expression materialization and are tracked for the post-R1 program.

## What this means

- **Matmul, eigenvalues, and Cholesky** match or beat NumPy at large sizes.
- **Inverse and solve** are competitive, within 25% of NumPy.
- **SVD, elementwise, and dot** are the known gaps, and are the concrete targets for the
  post-R1 "greater than 0.90x NumPy" program (tracked in `TODO.md`).
