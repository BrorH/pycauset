# Optimization Checklist & Roadmap

**Date:** December 10, 2025  
**Goal:** Achieve performance comparable to NumPy for small (in-memory) matrices and superior performance for large (out-of-core) matrices using PyCauset's Cooperative Compute Architecture.

## Context
We are systematically reviewing and optimizing the core matrix operations across all supported data types. The primary focus is on the CPU backend, ensuring that we leverage:
1.  **Streaming Micro-Kernels:** To handle out-of-core data efficiently without OS thrashing.
2.  **SIMD Instructions (AVX2/AVX-512):** For raw compute throughput.
3.  **Parallelism:** Effective multi-threading.

## Workflow per Item
For each entry below, the following steps must be completed:
1.  **Implementation Check:** Does the basic logic exist?
2.  **Correctness Verification:** Do unit tests pass?
3.  **Optimization:** Is it using the best algorithm (e.g., tiled/streaming for large data)?
4.  **Benchmark:** Run comparison vs NumPy (for small N) and PyCauset baseline (for large N).

---

## 1. Float64 (Double Precision)
*Priority: High (Current Focus)*

### Multiplication (`matmul`)
- [x] **Implementation:** `matmul_streaming_f64` added to `CpuSolver`.
- [x] **Correctness:** Verified with benchmarks.
- [x] **Optimization (Memory):** 
    - **Direct Path:** Implemented.
    - **Streaming:** Implemented.
    - **Anti-Nanny:** Implemented via `MemoryGovernor::should_use_direct_path`.
- [x] **Optimization (Compute):** OpenBLAS integration complete.
- [x] **Verification Status:**
    - [x] **In-Memory:** Verified (N=1k, 5k, 20k).
    - [x] **Out-of-Core:** Verified (N=40k, 36GB).
- [x] **Benchmark:** 
    - N=5,000: 1.03x speedup vs NumPy.
    - N=20,000: 0.97x speedup (Equal). Time reduced from 88s to 58s.

### Inversion (`inverse`)
- [x] **Implementation:** `inverse_direct` (LAPACK) added to `CpuSolver`.
- [x] **Correctness:** Verified with `benchmarks/benchmark_inverse_f64.py`. Fixed return-by-value issues using `inverse_to`.
- [x] **Optimization:** Uses `MemoryGovernor::should_use_direct_path` to select between LAPACK (RAM) and Block Gauss-Jordan (Disk).
- [ ] **Verification Status:**
    - [x] **In-Memory:** Verified (N=1k, 10k).
    - [ ] **Out-of-Core:** Pending verification.
- [x] **Benchmark:** 
    - N=1,000: ~1.4x speedup vs NumPy.
    - N=10,000: ~0.92x speedup (Comparable).
    - Validated correctness (A * A^-1 = I).

### Eigenvalues (`eigen`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Needs "Direct Path" logic.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy.

---

## 2. Float32 (Single Precision)
*Priority: High*

### Multiplication (`matmul`)
- [x] **Implementation:** Generalized `matmul_impl` handles Float32.
- [x] **Correctness:** Verified via `benchmarks/benchmark_float32_blas.py`.
- [x] **Optimization:** **Direct Path** and **Streaming** generalized. Added `beta=0` optimization. Added **Anti-Nanny** check to bypass streaming for RAM-resident matrices.
- [ ] **Verification Status:**
    - [x] **In-Memory:** Verified (N=6k).
    - [ ] **Out-of-Core:** Pending.
- [x] **Benchmark:** N=6000: 0.88x speedup vs NumPy. N=4096: Expected 1.00x with Anti-Nanny fix.

### Inversion (`inverse`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Check for out-of-core support.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy.

### Eigenvalues (`eigen`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Check for out-of-core support.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy.

---

## 3. Float16 (Half Precision)
*Priority: Medium (Specialized use cases)*

### Multiplication (`matmul`)
- [ ] **Implementation:** Check `matmul_impl` for Float16.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Ensure SIMD usage if available, or efficient emulation.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy (if supported) or PyTorch.

### Inversion (`inverse`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Stability concerns with low precision?
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs Baseline.

### Eigenvalues (`eigen`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Feasibility check.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs Baseline.

---

## 4. Integer (Int32/Int64)
*Priority: Medium*

### Multiplication (`matmul`)
- [ ] **Implementation:** Check existence.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Vectorization opportunities.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy.

### Inversion (`inverse`)
- [ ] **Implementation:** N/A (Usually results in floats).
- [ ] **Correctness:** N/A.
- [ ] **Optimization:** N/A.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** N/A.
    - [ ] **Out-of-Core:** N/A.
- [ ] **Benchmark:** N/A.

### Eigenvalues (`eigen`)
- [ ] **Implementation:** N/A (Usually results in floats/complex).
- [ ] **Correctness:** N/A.
- [ ] **Optimization:** N/A.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** N/A.
    - [ ] **Out-of-Core:** N/A.
- [ ] **Benchmark:** N/A.

---

## 5. Bit / Boolean
*Priority: Low (Specialized)*

### Multiplication (`matmul`)
- [ ] **Implementation:** Check `DenseBitMatrix`.
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Bit-packing and bitwise operations (popcount).
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs NumPy (packed).

### Inversion (`inverse`)
- [ ] **Implementation:** Check existence (GF(2) inversion?).
- [ ] **Correctness:** Verify with unit tests.
- [ ] **Optimization:** Method of Four Russians or similar?
- [ ] **Verification Status:**
    - [ ] **In-Memory:** Pending.
    - [ ] **Out-of-Core:** Pending.
- [ ] **Benchmark:** Compare vs Baseline.

### Eigenvalues (`eigen`)
- [ ] **Implementation:** N/A.
- [ ] **Correctness:** N/A.
- [ ] **Optimization:** N/A.
- [ ] **Verification Status:**
    - [ ] **In-Memory:** N/A.
    - [ ] **Out-of-Core:** N/A.
- [ ] **Benchmark:** N/A.

---

## Notes
*   **Complex Numbers:** Currently deferred.
*   **Benchmarks:** Always run `benchmarks/benchmark_cpu_vs_numpy.py` (or equivalent) after changes.
*   **Tests:** Ensure `tests/` cover edge cases (singular matrices, non-square, etc.).
