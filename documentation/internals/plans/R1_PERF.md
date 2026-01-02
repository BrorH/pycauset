# R1_PERF: Performance Optimization & Verification

## 1. The Problem
We have identified several "Performance/Verification Risks" that prevent us from claiming "NumPy Parity" or "Theoretical Optimality."

## 2. Objectives

### 2.1 The 50% Import Gap
*   **Symptom:** `import_matrix` runs at 2.5GB/s. Baseline `memcpy` is 5.0GB/s.
*   **Hypothesis:** OS Page Fault overhead + File System metadata updates during write.
*   **Action:**
    *   Profile with VTune/perf.
    *   Test `fallocate` / `SetFileValidData` to pre-allocate disk space.
    *   Test `MAP_POPULATE` (Linux) / `PrefetchVirtualMemory` (Windows).

### 2.2 "Fake" AVX-512
*   **Symptom:** `DenseBitMatrix` uses `std::popcount` (SWAR or generic instruction).
*   **Goal:** Use `_mm512_popcnt_epi64` (AVX-512 VPOPCNTDQ) where available.
*   **Action:**
    *   Add runtime dispatch for AVX-512.
    *   Implement AVX-512 kernels for `popcount`, `and`, `or`, `xor`.

### 2.3 Naive Threading
*   **Symptom:** `ParallelFor` splits work into `N / Threads` chunks. If one chunk hits a page fault, that thread stalls, and the whole op waits.
*   **Action:**
    *   Implement **Dynamic Scheduling** (Work Stealing or finer-grained chunks).
    *   Use `tbb::parallel_for` or a custom work-stealing queue if TBB is too heavy.

### 2.4 Pipeline Verification
*   **Symptom:** `AsyncStreamer` code *looks* correct, but we haven't *seen* the overlap.
*   **Action:**
    *   Instrument `AsyncStreamer` with NVTX ranges (NVIDIA Tools Extension).
    *   Visualize in Nsight Systems.
    *   Confirm `Compute` and `Transfer` bars overlap.

## 3. Deliverables
*   [ ] **Optimized Import:** >= 4.0GB/s (80% of baseline).
*   [ ] **AVX-512 BitMatrix:** Explicit intrinsics path.
*   [ ] **Robust Threading:** Dynamic scheduling in `ParallelUtils`.
*   [ ] **Pipeline Proof:** Nsight Systems screenshot/log confirming overlap.
