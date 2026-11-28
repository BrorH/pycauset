# pycauset.set_num_threads

Sets the number of threads to use for parallel operations.

## Syntax

```python
pycauset.set_num_threads(n)
```

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `n` | `int` | The number of threads to use. Must be $\ge 1$. |

## Description

One thread budget for the whole engine: `set_num_threads(n)` caps PyCauset's thread pool
(element-wise, reductions, structured kernels) **and** OpenBLAS (matrix products and every
LAPACK kernel: solve, inverse, eigen, SVD).

The default:

* `OPENBLAS_NUM_THREADS` or `OMP_NUM_THREADS` when set (then also the BLAS budget, as in an
  MPI / Slurm job with one thread per rank);
* otherwise the CPUs this process may run on (the affinity mask, `taskset`, not the host's
  count). LAPACK kernels then use at most 8 threads (more slow small factorizations down) and
  a matrix product up to 20.

A forked child (`multiprocessing` with the fork start method) builds its own pool.

## Example

```python
import pycauset
import os

# Use half the available cores
cores = os.cpu_count()
pycauset.set_num_threads(cores // 2)
```
