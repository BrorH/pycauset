# R1_PERF: Performance Optimization & Verification
**Status: Completed**

## 1. The Problem
We have identified several "Performance/Verification Risks" that prevent us from claiming "NumPy Parity" or "Theoretical Optimality."
Current benchmarks show `import_matrix` at ~50% of theoretical bandwidth, and threading stalls on page faults due to static partitioning.

**Note on "NumPy Parity":** We define parity as achieving **>0.90x** of NumPy's performance for I/O, data handling, and memory-resident operations. We are *not* currently targeting micro-optimization of heavy compute kernels (eig, matmul) beyond what BLAS/LAPACK provide, but we *must* ensure our data handling infrastructure (read/write/copy) does not introduce overhead.

## 2. Phased Implementation Plan

### Phase 1: Robust Threading (Dynamic Scheduling)
**Goal:** Eliminate stalls caused by page faults or uneven work distribution in `ParallelFor`.
*   **Context:** Current implementation uses static partitioning (`range / num_threads`). In an out-of-core system, if Thread 0 hits a hard page fault, Threads 1..N wait idly at the barrier.
*   **Deliverables:**
    *   [x] **Work-Stealing Queue / Dynamic Chunking:** Refactor `ParallelUtils.cpp` to use a dynamic task queue or atomic index for chunk claiming.
    *   [x] **Grain Size Tuning:** Implement heuristics to determine optimal chunk size based on L3 cache size (not just thread count).
    *   [x] **Verification:** Micro-benchmark showing linear scaling even with induced delays (simulated page faults) in single threads.

### Phase 2: IO & Memory Throughput (The Import Gap)
**Goal:** Increase `import_matrix` throughput from 2.5GB/s to >4.0GB/s (80% of memcpy baseline) and ensure efficient reads.
*   **Context:** Writing new files triggers zero-filling security features in the OS (slowing down import). Reading triggers page faults.
*   **Deliverables:**
    *   [x] **Pre-allocation (Windows):** Implement `SetFileValidData` in `MemoryMapper` to bypass zero-filling for new files (requires `SE_MANAGE_VOLUME_NAME` privilege handling).
    *   [x] **Pre-allocation (Linux):** Implement `fallocate` to reserve disk blocks.
    *   [x] **Bulk Paging (Read Optimization):** Verify and tune `PrefetchVirtualMemory` (Windows) and `MAP_POPULATE` (Linux) usage in `IOAccelerator` to ensure we are saturating read bandwidth.
    *   [x] **Huge Pages Investigation:** Evaluate `MADV_HUGEPAGE` / Large Pages for reducing TLB misses on >10GB matrices.

### Phase 3: Data Handling Micro-optimizations (AVX-512)
**Goal:** Achieve theoretical optimality for `DenseBitMatrix` operations (fundamental Causal Set data structure).
*   **Context:** Current `std::popcount` is good but not optimal for large bitstreams. AVX-512 `VPOPCNTDQ` can process 512 bits per cycle.
*   **Deliverables:**
    *   [x] **Alignment Check:** Verify `MemoryMapper` provides 64-byte alignment (critical for AVX-512).
    *   [x] **Runtime Dispatch:** Implement `CpuId` check to safely select AVX-512 paths at runtime.
    *   [x] **Intrinsics Kernels:** Implement `_mm512_popcnt_epi64` and bitwise logic (`and`, `or`, `xor`, `not`) kernels.

### Phase 4: Pipeline Verification & Direct Path
**Goal:** Prove that the "Cooperative Architecture" works as intended and that the "Direct Path" bypasses overhead for RAM-resident data.
*   **Context:** `AsyncStreamer` logic exists but hasn't been visually verified. `MemoryGovernor` has logic for "Direct Path" (bypassing tiling for RAM-resident data), which is critical for NumPy parity.
*   **Deliverables:**
    *   [x] **Direct Path Verification:** Verify `MemoryGovernor::should_use_direct_path` correctly routes in-memory workloads to the low-overhead path (matching NumPy speed).
    *   [x] **NVTX Instrumentation:** Add NVIDIA Tools Extension markers to `AsyncStreamer`, `Compute`, and `Transfer` phases.
    *   [x] **Nsight Systems Validation:** Capture a trace confirming overlap.
    *   [x] **Benchmark Suite Update:** Add `benchmarks/benchmark_io_throughput.py` and `benchmarks/benchmark_threading_stress.py` to CI.

### Phase 5: Validation, Documentation & Cleanup
**Goal:** Ensure the new performance infrastructure is robust, documented, and the codebase is clean of legacy implementations.
*   **Context:** We are replacing core infrastructure (threading, IO). We must verify correctness under stress and document the new architecture. "Deprecation" means complete removal ("Purge").
*   **Deliverables:**
    *   [x] **Extensive Test Suite (Correctness First):**
        *   `tests/cpp/test_parallel_utils.cpp`: Verify dynamic scheduling edge cases (exceptions, uneven workloads, single-thread fallback, 0-range).
        *   `tests/test_io_accelerator.cpp`: Verify prefetch/discard behavior on Windows/Linux (mocked if necessary).
        *   `tests/python/test_io_consistency.py`: Verify data integrity after high-speed imports and "Direct Path" operations.
    *   [x] **Benchmark Suite (Performance Verification):**
        *   `benchmarks/benchmark_io_throughput.py`: Measure Read/Write bandwidth vs `memcpy`.
        *   `benchmarks/benchmark_stress.py`: Measure scaling efficiency and stall resistance.
        *   `benchmarks/benchmark_numpy_parity.py`: Verify >0.90x parity for IO and memory-resident ops.
    *   [x] **Documentation (per Protocol):**
        *   **Internals:** Update `internals/Compute Architecture.md` with the new Dynamic Scheduling model.
        *   **Internals:** Update `internals/MemoryArchitecture.md` with `SetFileValidData` / `PrefetchVirtualMemory` details.
        *   **API:** Document any new tuning knobs in `docs/parameters/`.
    *   [x] **Cleanup (Purge):**
        *   Remove all traces of static partitioning from `ParallelUtils`.
        *   Remove legacy "naive" file writing paths in `MemoryMapper`.
        *   Ensure no dead code remains from the old IO strategies.

### Phase 6: NumPy Parity & BitMatrix Optimization (Completed Jan 2026)
**Goal:** Achieve >0.90x performance parity with NumPy for all IO operations and optimize BitMatrix operations.
*   **Context:** Initial benchmarks showed poor read performance due to Python iteration overhead. BitMatrix operations were slow due to lack of SIMD.
*   **Deliverables:**
    *   [x] **Zero-Copy IO:** Implemented `_to_numpy_fast` using `memcpy` for Float64/Int32, achieving >3x NumPy speed.
    *   [x] **Optimized Complex IO:** Implemented specialized C++ loops for Complex128, achieving >3.5x NumPy speed.
    *   [x] **SIMD Bit Packing:** Implemented SSE2 intrinsics for `DenseBitMatrix` write (packing), achieving 5x speedup.
    *   [x] **SIMD Bit Unpacking:** Implemented SSE2 intrinsics for `DenseBitMatrix` read (unpacking), achieving 1.9x speedup.
    *   [x] **Bitwise Operations:** Implemented SIMD-accelerated `__xor__`, `__and__`, `__or__` for `DenseBitMatrix`, achieving ~5x speedup vs NumPy.
    *   [x] **Documentation:** Created `documentation/guides/performance.md` detailing these wins.

## 3. Success Criteria
*   **Import Speed:** > 4.0 GB/s on NVMe (Write).
*   **Read Speed:** > 0.90x NumPy `mmap` load speed.
*   **BitMatrix Popcount:** > 2x speedup over `std::popcount` on AVX-512 hardware.
*   **Threading:** No stalls observed in `benchmark_stress.py` when memory pressure is high.
*   **Pipeline:** Visual confirmation of overlap in Nsight Systems.
