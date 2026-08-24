# PyCauset CPU/GPU Optimization Status

**Last verified:** against `src/compute/cpu/CpuSolver.cpp`, `src/accelerators/cuda/CudaDevice.cu`, and `src/accelerators/cuda/CudaSolver.cu` (current HEAD).
**Companion to:** `SUPPORT_READINESS_FRAMEWORK.md` §2.2.1 (the routing-policy handoff table): this file explains *why* each op is fast or slow and what to do next.

> **Scale assumption:** we will run on "absurdly large" matrices. Everything below is graded against two thresholds:
> 1. **Correctness at scale**: no silent wrong answers, no int32 index overflow, no naive O(n³) scalar kernels that take days.
> 2. **Throughput**: BLAS/LAPACK/cuBLAS/cuSOLVER where the math is BLAS-shaped; SIMD/threading where it is elementwise.

---

## 1. Executive summary

**The R1 gate is correctness + no silent wrong answers + explicit support status: not max speed.** This report is the map for the *continuous post-R1 performance program*, and it flags the three places that would bite us *first* on huge matrices:

1. **`solve` (dense double) and `lu` (dense double) are naive scalar Gaussian elimination**: O(n³) single-threaded scalar, *not* LAPACK `dgesv`/`dgetrf`. This is the single biggest CPU gap. (Float paths already use Eigen/LAPACK: the inconsistency itself is a bug.)
2. **Bundled OpenBLAS is LP64 (32-bit `lapack_int`), ABI-consistent: verified.** The DLL exports `dgemm_`/`dgesv_`/`dgetrf_` (no `*_64_` ILP64 symbols), matching the code's `lapack_int = int`. This is *correct* as-is; the stale `// likely ILP64` comment in `CpuSolver.cpp` was removed. The int32 LAPACK indices are **per-dimension**, so they only overflow near n ≈ 2.1e9: memory-unreachable for dense matrices. **The real "absurdly large" ceiling is RAM/VRAM, not int32**: that is the out-of-core/streaming problem below, not an index-ABI problem.
3. **GPU is far from parity**: only `matmul`, `inverse`, `cholesky`, `add`, `subtract`, `multiply_scalar`, `batch_gemv`, and `solve` are GPU-implemented. Everything else is a CPU-route stub (throws → falls back). No GPU `qr`/`svd`/`eig`/`lu` yet.

**What is already fast (do not re-do):** `matmul` (OpenBLAS/cuBLAS + AVX-512 bit path), `inverse`/`cholesky`/`qr`/`svd` (LAPACK), elementwise `add/sub/mul/div` (AVX2 runtime-dispatched + thread pool), reductions (`sum`/`dot`/`frobenius_norm`, OpenMP).

---

## 2. Per-operation matrix

Legend:
- **CPU**: backend actually used on CPU. ✅ = optimized library call; ⚠️ = scalar/naive or single-threaded.
- **GPU**: ✅ = implemented kernel; ❌ = stub that throws → falls back to CPU (this is "CPU-route" in the SRP table).
- **Stream**: whether an out-of-core/VRAM-chunking path exists (`naive` = must materialize the whole result in RAM/VRAM at once).

| Op | CPU backend | GPU | Stream | Scale grade |
|---|---|---|---|---|
| `matmul` | OpenBLAS `cblas_dgemm`/`sgemm` (multithreaded); bit×bit → AVX-512 popcount (cpuid-guarded) | ✅ cuBLAS (dense f32/f64; bit→int32) | ✅ VRAM chunking (`matmul_streaming`, `gemm_streaming`) | ✅ |
| `inverse` | LAPACK `dgetrf`+`dgetri` | ✅ cuSOLVER `getrf`/`getri` (+ `inverse_incore`) | ❌ naive | ✅ |
| `solve` | ✅ LAPACK `dgesv` (float: Eigen `PartialPivLU`) | ✅ custom LU row-panel / forward / back-sub | ❌ naive | ✅ (was naive double GE; fixed) |
| `lu` | ✅ LAPACK `dgetrf` (float: `sgetrf`) | ❌ stub | ❌ naive | ✅ (was naive double GE; fixed) |
| `cholesky` | LAPACK `dpotrf`/`spotrf` | ✅ custom kernel (cuSOLVER path) | ❌ naive | ✅ |
| `qr` | LAPACK `dgeqrf`+`dorgqr` | ❌ stub | ❌ naive | ✅ (add cuSOLVER later) |
| `svd` | LAPACK `dgesvd`/`sgesvd` (thin) | ❌ stub | ❌ naive | ✅ (add `gesvd` later) |
| `eigh` / `eigvalsh` | LAPACK `dsyev` (C++); NumPy fallback (Python) | ❌ (only `eigvals_arnoldi` is GPU) | ❌ naive | ⚠️ dense eig is O(n³)/O(n²) mem: use Arnoldi for huge |
| `eigvals_arnoldi` | Eigen `EigenSolver` on small Hessenberg | ✅ Arnoldi driver | ✅ iterative (k-step) | ✅ |
| `add` / `subtract` | AVX2 (runtime-dispatched) + thread pool; **f64 sub = scalar** | ✅ custom kernels | ❌ naive | ✅ (f64 sub gap) |
| `elementwise_multiply` / `_divide` | AVX2 f32; **f64 = scalar** | ❌ stub | ❌ naive | ⚠️ f64 SIMD + GPU gap |
| `multiply_scalar` | AVX2 f32 + thread pool; f64 scalar | ✅ custom kernel | ❌ naive | ✅ |
| `dot` / `dot_complex` | OpenMP reduction (real + complex) | ❌ stub | ❌ naive | ✅ |
| `sum` (vec & mat) | OpenMP reduction | ❌ stub | ❌ naive | ✅ |
| `trace` | scalar O(n) (structured shortcuts) | ❌ stub | ❌ naive | ✅ |
| `determinant` | structured shortcuts; general = Eigen `PartialPivLU` (single-thread) | ❌ stub | ❌ naive | ⚠️ O(n³) single-thread |
| `frobenius_norm` | OpenMP reduction | ❌ stub | ❌ naive | ✅ |
| `batch_gemv` | CPU (ParallelFor) | ✅ cuBLAS gemv (+ streaming) | ✅ VRAM chunking | ✅ |
| `matrix_vector_multiply` / `vector_matrix_multiply` | ParallelFor | ❌ stub | ❌ naive | ⚠️ should be BLAS `gemv` |
| `outer_product` | ParallelFor | ❌ stub | ❌ naive | ⚠️ should be BLAS `ger` |
| vector add/sub/scalar/cross | ParallelFor | ❌ stub | ❌ naive | ✅ |

Python-level linalg endpoints (`solve`, `lstsq`, `slogdet`, `cond`, `eigh`, `eigvalsh`) currently use **NumPy fallback or composition**: see `SUPPORT_READINESS_FRAMEWORK.md` §2.2.2. `solve_triangular` / `lu` / `cholesky` / `svd` / `pinv` are still `blocked` at the Python endpoint layer.

---

## 3. CPU backend detail (what is actually there)

| Mechanism | Where | Notes |
|---|---|---|
| **OpenBLAS** (`cblas_dgemm`/`sgemm`) | `CpuSolver::matmul`, `matmul_dense`, bit→int accumulation | Multithreaded; the workhorse for dense GEMM. |
| **LAPACK** (`dgetrf`/`dgetri`, `dpotrf`, `dgeqrf`/`dorgqr`, `dgesvd`, `dsyev`) | inverse, cholesky, qr, svd, eigh | Real float/double only in most paths; complex routed to NumPy fallback at Python level. |
| **AVX2 elementwise** (runtime-dispatched via `has_avx2()` cpuid, scalar fallback) | `avx2_add/sub/mul/div_f32`, `avx2_add_f64` | f32 fully covered; f64 only `add`. |
| **AVX-512 popcount** (cpuid `has_avx512_vpopcntdq` + `vpopcntdq`; `target` attr on GCC/Clang) | bit-matmul `dot_product_avx512` | Runtime-dispatched; baseline-safe. |
| **`ParallelFor` thread pool** | every `CpuSolver::*` elementwise/vector loop | Custom global ThreadPool. |
| **OpenMP reductions** | `sum`, `dot`, `dot_complex`, `frobenius_norm` | `#pragma omp parallel for reduction`. |
| **Eigen** | determinant (PartialPivLU), float solve, Arnoldi small Hessenberg | Single-threaded; fine for small/structured, not for huge dense. |

### 3.1 Dense float64 `solve`/`lu` now use LAPACK (fixed)

~~The two CPU inconsistencies~~ are resolved: both dense `float64` paths now go through LAPACK `dgesv`/`dgetrf`, matching the float paths (Eigen `PartialPivLU` / `sgetrf`). The `P@L@U` reconstruction convention (column-index P built from `ipiv`) was empirically verified against NumPy, and the paths are guarded by `tests/python/test_edge_cases_core.py::TestDenseFactorizationsLapack`.

---

## 4. GPU backend detail (what is actually there)

**Implemented (in `CudaDevice.cu` / `CudaSolver.cu`):**

| Op | GPU backend |
|---|---|
| `matmul` | cuBLAS `gemm` (f32/f64); bit×bit→int32 custom; streaming variants (`matmul_streaming`, `gemm_streaming`) chunk on available VRAM |
| `inverse` / `inverse_incore` | cuSOLVER `getrf`+`getri` (in-core and out-of-core variants) |
| `cholesky` | cuSOLVER/custom potrf |
| `add` / `subtract` / `multiply_scalar` | custom elementwise kernels |
| `batch_gemv` (+ `batch_gemv_streaming`) | cuBLAS batched GEMV, VRAM-chunked |
| `solve` | custom LU row-panel + forward/back substitution |
| `eigvals_arnoldi` | Arnoldi iteration driver (k eigenpairs) |
| `matmul_bit` | bit-matrix GEMM (bool → int32 popcount) |

**Stubbed → CPU fallback (throws "not implemented"):** `sum` (vec+mat), `trace`, `determinant`, `qr`, `dot`, `dot_complex`, `add_vector`, `subtract_vector`, `scalar_multiply_vector`, `scalar_multiply_vector_complex`, `scalar_add_vector`, `cross_product`, `matrix_vector_multiply`, `vector_matrix_multiply`, `outer_product`, `elementwise_multiply`, `elementwise_divide`, `compute_k_matrix`, `frobenius_norm`.

> Note: `dot_complex` and `scalar_multiply_vector_complex` are **stubs**, not implemented: this corrects an earlier survey that mislabeled them.

**CUDA build is currently `OFF`** on this machine (GTX 1060, Pascal CC 6.1): CUDA 13.0 dropped Pascal (`--list-gpu-arch` starts at `compute_75`), and CUDA 12.6 (which still supports Pascal) cannot compile against VS 2026 (`cudafe++` access violation even with `-allow-unsupported-compiler`; no CUDA VS integration for VS 2026). **Unblock requires a VS 2022 (MSVC 14.4x) install.** The Python `cuda.is_available()` is therefore `False` until `ENABLE_CUDA=ON` builds.

---

## 5. Correctness / scale risks (must-fix before "disgustingly large" claims)

1. **LAPACK/BLAS int32 indexing is a non-issue for dense matrices.** `lapack_int`/`blasint` are int32 but are used for *single dimensions* (`n`, `lda`, `M/N/K`), which overflow only near n ≈ 2.1e9: unreachable for dense (RAM-bound first). **Verified:** bundled OpenBLAS is LP64 and ABI-consistent with the code. The real scale limit is RAM/VRAM → out-of-core (§6 items 6-7). Guard against it by keeping all *flattened element* loops on `size_t`/`uint64_t` (already the case) and never casting a total-element count to `int`.
2. ~~`solve`/`lu` double naive kernels~~ → **FIXED** (LAPACK `dgesv`/`dgetrf`, §3.1). Singularity now matches LAPACK/NumPy exactly-zero-pivot semantics.
3. ~~AVX-512 `/arch` leakage~~ → **FIXED**. SIMD kernels are runtime-dispatched via cpuid (`has_avx512_vpopcntdq()` / `has_avx2()`) with scalar fallbacks; `-march=native` was removed and the AVX2 kernels now carry per-function `__attribute__((target("avx2")))` (MSVC emits intrinsics without `/arch`, so no flag is needed or leaked there). Distribution binaries are baseline-safe.
4. **f64 elementwise is largely scalar** (`sub`/`mul`/`div` in double fall to `scalar_*`). Not wrong, just slower; low risk.
5. **GPU `CMAKE_CUDA_ARCHITECTURES "native"`**: fine locally, wrong for distribution (would compile only for the build machine's arch). Replace with an explicit arch list (e.g., `50;60;61;70;75;80;86;90` + PTX) for wheels.

---

## 6. Forward catalog: "potential other ways"

Ordered by expected payoff for huge matrices:

1. **Full cuBLAS/cuSOLVER wiring** for the missing factorizations: `qr` (`geqrf`+`orgqr`), `svd` (`gesvd`), `lu` (`getrf`), `eig` (`syevd`), plus batched `getri` for inverse-of-many. One shared `CudaLinalg` dispatch layer instead of ad-hoc kernels.
2. ~~LAPACK `dgesv`/`dgetrf` for double `solve`/`lu`~~ → **DONE** (verified against NumPy).
3. **64-bit indexing already holds where it matters.** Flattened element loops use `size_t`/`uint64_t`, and the memory-mapped containers use 64-bit offsets; LAPACK/BLAS int32 is per-dimension and safe for any dense size that fits in RAM. Keep this invariant (never cast element counts to `int`). ILP64 is only worth revisiting if we ever support single-dimension n > 2.1e9 (banded/sparse extremes): not on the dense roadmap.
4. **Runtime SIMD dispatch completion.** Baseline-safety is done (`-march=native` removed; AVX2/AVX-512 kernels carry per-function `target` attributes and are cpuid-gated). Remaining throughput work: (a) f64 sub/mul/div SIMD, (b) AVX-512 for elementwise and reductions.
5. **BLAS batched ops** (`cblas_*_batch`, cuBLAS batched GEMM/GEMV) for ensembles of many small causal-set matrices: the causal-set workload that does not fit the single-huge-matrix pattern.
6. **Tiled/blocked out-of-core executor.** Generalize the existing GPU VRAM chunking to a generic "tile + accumulator" loop keyed on the `MemoryGovernor` budget, so `add`/`subtract`/`inverse`/`qr`/`svd` can run on memory-mapped `.pycauset` containers without materializing the full result in RAM. Today only `matmul`/`batch_gemv` stream.
7. **Mixed CPU+GPU tandem** (cooperative execution): split a tile loop across OpenBLAS (CPU) and cuBLAS (GPU) simultaneously. Already documented as a future direction in `SUPPORT_READINESS_FRAMEWORK.md` §2.2.1.
8. **Mixed-precision iterative refinement** (fp32 factorization + fp64 residual correction) to halve memory/time for `solve`/`inverse` on huge matrices, with a documented accuracy contract.
9. **fp16/bf16 storage** only where memory-bound; note the GTX 1060 (Pascal) has **no Tensor Cores**, so fp16 GEMM is not the win it would be on Volta+.
10. **Thread hygiene**: OpenMP (`OMP_NUM_THREADS`) and the custom `ParallelFor` pool currently coexist: pinning + a single shared scheduler avoids oversubscription when both run.

---

## 7. Priority action list (next)

| # | Action | Why now | Effort |
|---|---|---|---|
| 1 | ~~Fix double solve + lu → LAPACK~~ → DONE (`dgesv`/`dgetrf`, verified) | Biggest CPU gap; silent-perf inconsistency | S |
| 2 | ~~Verify OpenBLAS ILP64~~ → DONE: LP64, ABI-consistent; removed stale comment | Confirmed no index-ABI risk | S |
| 3 | Unblock CUDA build (VS 2022) and re-enable `ENABLE_CUDA` | Gates all GPU work | User action + M |
| 4 | ~~Scope AVX-512 TU isolation / runtime dispatch~~ → DONE (cpuid-gated + per-fn `target`, `-march=native` removed) | Top production crash risk | M |
| 5 | cuBLAS/cuSOLVER for `qr`/`svd`/`lu`/`eig` | GPU parity | M-L |
| 6 | f64 elementwise SIMD (sub/mul/div) | Cheap throughput win | S |
| 7 | Generic tiled out-of-core executor | "Absurdly large" RAM ceiling | L |

Companion tables: `SUPPORT_READINESS_FRAMEWORK.md` §2.2.1 (routing policy), `R1_EXECUTION.md` (release gate), `TODO.md` (roadmap).
