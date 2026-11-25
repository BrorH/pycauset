# Performance Guide

PyCauset runs on the CPU: OpenMP for element-wise kernels, OpenBLAS/LAPACK for products,
solves and decompositions. This page covers what is fast, what to avoid, and the few
settings that matter. (A CUDA backend exists in `experimental/cuda` but is not built or
supported.)

## Threads

NumPy brings its own OpenBLAS. By default each thread pool (OpenMP, PyCauset's OpenBLAS,
NumPy's OpenBLAS) keeps its threads spinning after a call (OpenBLAS for about 0.1 s), which
starves whichever pool runs next: alternating NumPy and PyCauset calls made either one up
to ~3× slower.

`import pycauset` therefore sets two defaults before any BLAS library loads (values you set
yourself are kept):

| variable | default set by PyCauset | effect |
|---|---|---|
| `OMP_WAIT_POLICY` | `PASSIVE` | OpenMP threads sleep instead of spinning between kernels |
| `OPENBLAS_THREAD_TIMEOUT` | `4` | OpenBLAS threads stop spinning almost immediately |

NumPy's OpenBLAS reads the variable when NumPy is first imported, so **import pycauset
before numpy** (or export `OPENBLAS_THREAD_TIMEOUT=4` in your shell) to cover NumPy's
threads as well.

## How fast, compared with NumPy

Measured on a 4-vCPU Linux machine (`benchmarks/r2_parity.py`, float64, N = 1024; ratios
vary by about ±10% between runs):

| operation | PyCauset vs NumPy |
|---|---|
| `A @ B`, `solve`, `eigh`, `dot`, export to NumPy | at parity (0.9–1.1×) |
| `A + B`, `A * B`, `A + s` | at parity at N = 1024; 2–3× faster at N = 3000 (multithreaded) |
| `inv`, `det` | 0.7–1.0× |

Where PyCauset is structurally ahead:

* **Causal (bit) matrices.** A causal matrix stores 1 bit per relation: 8× less than a
  NumPy `bool` array, 64× less than float64. Products of bit matrices use word-level
  popcount kernels (n = 2000: 0.05 s, versus 0.11 s for a float64 BLAS product and seconds
  for NumPy's `bool @ bool`). `pc.sum(C)` counts the relations from the bits (n = 20,000
  in 2 ms).
* **Structured results.** `C + C.T` and `C - C.T` are read from `C`'s bits without being
  written out, and iterative eigensolvers work on them directly: the top of the spectrum of
  `C - C.T` for n = 20,000 in about 2 s and 100 MB. See the
  [[guides/Lazy Evaluation|Lazy Evaluation guide]].
* **Exact integers.** Integer matrix products run exactly through float64 BLAS when the
  values provably fit (50× faster than an integer kernel); sums and dot products of integer
  arrays are exact Python ints.
* **Larger than RAM.** Matrices above the memory threshold are file-backed (memory-mapped)
  and operations work on them out of core.

## Memory: RAM first, then disk

Matrices live in RAM until their size passes the memory threshold; larger ones are backed by
files in the backing directory and memory-mapped.

```python
import pycauset as pc

pc.set_backing_dir("./storage")                # where file-backed matrices go
pc.set_memory_threshold(256 * 1024**2)         # bytes; larger matrices use files
```

Operations on file-backed matrices stream through RAM in tiles; the threshold for that is
`pc.set_io_streaming_threshold(bytes)`.

## Things to avoid

* **Element-by-element loops in Python** (`for i ...: A[i, j]`): each access is a Python
  call. Use whole-matrix operations, slicing, or `np.asarray(A)` once.
* **Converting back and forth with NumPy inside a loop**: each `np.asarray(A)` copies the
  matrix.

## See also

- [[docs/functions/pycauset.set_backing_dir.md|pycauset.set_backing_dir]]
- [[docs/functions/pycauset.set_memory_threshold.md|pycauset.set_memory_threshold]]
- [[docs/functions/pycauset.set_io_streaming_threshold.md|pycauset.set_io_streaming_threshold]]
- [[guides/Lazy Evaluation|Lazy Evaluation]]
- [[internals/Streaming Manager.md|Streaming Manager]]
