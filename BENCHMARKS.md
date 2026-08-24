# PyCauset Benchmarks

Reproducible benchmark results for PyCauset's core linear-algebra surface, measured
against NumPy. Run them yourself with:

```
python benchmarks/bench.py
```

## Methodology

- **Hardware:** Intel Core i9-10850K (10 cores @ 3.6 GHz), 32 GB RAM, Windows 11.
- **Versions:** NumPy 2.3.5 (OpenBLAS), PyCauset 0.5.1 (OpenBLAS 0.3.26).
- **Timing:** `time.perf_counter`, best-of-N per operation. Dense `float64` unless noted.
- **Ratio:** `numpy_time / pycauset_time` — values **above 1.0x mean PyCauset is faster**.

> **Honest framing.** PyCauset's dense kernels are built on the *same* OpenBLAS/LAPACK
> backend as NumPy, so the realistic goal is **parity**, not an order-of-magnitude
> speedup. PyCauset's differentiation is elsewhere: **bit-packed storage (8x smaller)**,
> **memory-mapped / out-of-core matrices**, and **lazy expression fusion**. The
> "≥0.90× NumPy everywhere" target is tracked in `TODO.md` under the post-R1 program.

## Dense float64 matmul (C = A @ B)

| n | NumPy | PyCauset | ratio |
|---|---|---|---|
| 1000 | 6.0ms | 10.3ms | 0.58x |
| 2000 | 60.0ms | 70.5ms | 0.85x |
| 4000 | 493.7ms | 492.9ms | 1.00x |
| 6000 | 1.60s | 1.50s | **1.07x** |
| 8000 | 3.74s | 3.51s | **1.07x** |

Matmul matches NumPy at n = 4000 and **pulls ahead (1.07x faster) at n = 6000+**. Small
sizes carry fixed Python-dispatch overhead that amortizes away as the matrix grows. (At
n = 8000 the operands exceed the 1 GB RAM budget and PyCauset transparently memory-maps to
disk, yet still finishes faster than NumPy.)

## Dense float64 factorizations (LAPACK vs NumPy)

| op | n | NumPy | PyCauset | ratio |
|---|---|---|---|---|
| inverse | 500 | 9.8ms | 9.4ms | 1.05x |
| inverse | 1000 | 35.1ms | 62.4ms | 0.56x |
| inverse | 2000 | 207.7ms | 278.0ms | 0.75x |
| cholesky | 500 | 1.9ms | 2.7ms | 0.71x |
| cholesky | 1000 | 10.2ms | 10.8ms | 0.95x |
| cholesky | 2000 | 54.8ms | 42.7ms | **1.28x** |
| solve | 500 | 2.4ms | 4.4ms | 0.56x |
| solve | 1000 | 17.0ms | 23.2ms | 0.73x |
| solve | 2000 | 110.0ms | 147.5ms | 0.75x |

Factorizations are **competitive** (same order of magnitude as NumPy); Cholesky is already
**faster than NumPy at n = 2000**. `solve`/`inverse` are the nearest post-R1 optimization
targets to reach 0.90x.

## Bit-packed boolean matrices (causal-set storage)

| metric | NumPy bool | PyCauset bit | reduction |
|---|---|---|---|
| storage (10000x10000) | 100.0 MB | 12.5 MB | **8x** |
| bit matmul (10000x10000) | — | 2.55s | AVX-512 popcount |
| storage (20000x20000) | 400.0 MB | 50.0 MB | **8x** |
| bit matmul (20000x20000) | — | 18.8s | AVX-512 popcount |

Causal-set matrices are boolean and are stored **bit-packed**: 8x smaller than a `bool`
array, with an AVX-512-accelerated multiplication. A 10000x10000 bit matmul completes in
~2.5s; the equivalent NumPy `bool` matmul does not use packed bit arithmetic and is far
slower.

## Out-of-core (memory-mapped) matrices

PyCauset spills matrices to disk when they exceed the RAM budget (`set_memory_threshold`),
so a computation can proceed on data that does not fit in memory — a regime where NumPy
raises `MemoryError`.

Verified demonstration (72 MB matrix forced to disk with a 1 MB budget):

| step | result |
|---|---|
| `FloatMatrix(3000)` with 1 MB threshold | 10 ms; backing file created on disk (`.pycauset/*.tmp`) |
| `trace(identity)` | 1.9 ms, returns 3000 (correct) |
| `to_numpy(..., allow_huge=True)` | 12 ms, diagonal sum 3000 (correct) |

The same pattern scales to matrices larger than physical RAM; the "humongous" scripts in
`benchmarks/` (e.g. `benchmark_humongous.py`) exercise a 50 GB inverse end-to-end.

## What this means for large workloads

- For **dense** linear algebra, PyCauset matches NumPy (same BLAS backend) and edges ahead
  on large matmul (1.07x at n = 6000+).
- For **causal-set** workloads, the bit-packed representation is the win: 8x memory
  reduction and hardware-popcount multiplication.
- For **out-of-core** workloads, PyCauset matrices can be memory-mapped so a computation
  can proceed on data that does not fit in RAM — a regime where NumPy raises
  `MemoryError`. (Out-of-core *performance* validation is part of the post-R1 program.)
