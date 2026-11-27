# Compute Architecture

How an operation reaches a kernel. R1 runs on the CPU only; the CUDA backend is parked in
`experimental/cuda` (its README lists the dormant hooks still in the core).

## The path of an operation

```mermaid
flowchart LR
    Py["Python entry function<br/>(pycauset/_internal/ops.py)"] --> Bind["binding<br/>(src/bindings)"]
    Bind --> Entry["engine entry function<br/>(structure rules, normaliser)"]
    Entry --> Ctx["ComputeContext::get_device()"]
    Ctx --> Auto["AutoSolver"]
    Auto --> Cpu["CpuDevice"]
    Cpu -->|"fits in RAM"| Solver["CpuSolver kernels"]
    Cpu -->|"out of core"| SM["StreamingManager"]
    SM --> Worker["CpuComputeWorker<br/>(tile kernels)"]
    Worker --> Solver
```

- **Entry function.** Each operation has one: it applies the structure rules (which class
  the result is), the normaliser (views read in place) and allocates the result. The
  procedure is [[project/protocols/Adding Operations|Adding Operations]].
- **`ComputeContext`** (`src/compute/ComputeContext.cpp`): a singleton whose device is an
  `AutoSolver`.
- **`AutoSolver`** (`src/compute/AutoSolver.cpp`): the device router. With no GPU device
  loaded (always, in R1) every call goes to `CpuDevice`. Its GPU routing, cost model and
  pinned-memory calls are dormant CUDA hooks, kept by design (review/DEADCODE.md).
- **`CpuDevice`** (`src/compute/cpu/CpuDevice.cpp`): picks in-memory or out-of-core
  execution. Matmul and element-wise operations stream when the operands and result do not
  fit (`out_of_core`); everything else runs in memory.
- **`CpuSolver`**: the kernels, split by family:

| File | Kernels |
|---|---|
| `CpuMatmul.cpp` | dense GEMM (BLAS), structured and bit matmul (`popcount`) |
| `CpuElementwise.cpp` | `+ - * /`, bitwise ops, packed structured element-wise |
| `CpuLinalg.cpp` | LAPACK factorizations and eigensolvers (`eigh`, `eig`, `cholesky`, `lu`, `qr`) |
| `CpuSolver.cpp` | dot, norms, matrix-vector, `compute_k`, `svd`, `solve` |
| `CpuIterative.cpp` | Arnoldi, skew eigensolvers, batched matrix-vector |
| `CpuKernelsCommon.hpp` | shared helpers |

## Out-of-core execution

`StreamingManager` (`src/compute/StreamingManager.cpp`) tiles a matmul or element-wise
operation to fit the `MemoryGovernor` budget and hands each tile to a `ComputeWorker`
(`include/pycauset/compute/ComputeWorker.hpp`: `matmul_tile`, `elementwise_tile`). The only
worker is `CpuComputeWorker`, which calls the in-memory kernels on the tile. Details:
[[internals/Streaming Manager|Streaming Manager]].

## LAPACK

Factorizations and eigen problems call LAPACKE in the operand's precision (single stays
single, as in NumPy) and need the matrix in memory:

| Operation | LAPACK |
|---|---|
| `eigh`, `eigvalsh` | `?syevd` / `?heevd` |
| `eig`, `eigvals` | `?geev` |
| `svd` | `?gesdd` |
| `solve`, `invert` | `?gesv` (`invert` solves `A X = I`, as NumPy does) |
| `lu`, `determinant`, `slogdet` | `?getrf` |
| `cholesky` | `?potrf` |
| `qr` | `?geqrf` + `?orgqr` / `?ungqr` |
| `pinv`, `lstsq` | via `svd` (`lstsq` is `pinv(a) @ b`) |

Structure shortcuts come first: the determinant of a triangular or diagonal matrix is the
product of its diagonal, an identity's is `scale^n`, and so on.

`eigvals_arnoldi(a, k, m, tol)` needs only matrix-vector products, so it works on
file-backed matrices that do not fit in RAM.

## Parallelism and SIMD

- **`ParallelFor`** (`include/pycauset/core/ParallelUtils.hpp`): a thread pool with dynamic
  chunks; threads take `range / (threads * 4)`-sized chunks from an atomic counter, so one
  thread stalled on a page fault does not hold the others. `pc.set_num_threads` caps it.
- **SIMD:** dense float32 / float64 `+ - * /` have AVX2 kernels, chosen at run time
  (`cpuid`); other dtypes use compiler-vectorised loops. Bit matrices use hardware
  `popcount`.

## Seeing what ran

- `pc.last_io_trace(op)`: the kernels an operation ran, its operands and whether it
  streamed ([[internals/Streaming Manager|Streaming Manager]]).
- `pc._debug_last_kernel_trace()` / `pc._debug_clear_kernel_trace()`: the engine's kernel
  log (thread-local), used by the tests and the operation atlas.

## See also

- [[internals/Streaming Manager|Streaming Manager]]
- [[internals/Memory and Data|Memory and Data]]
- [[internals/Algorithms|Algorithms]]
