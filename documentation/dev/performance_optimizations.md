# Performance Optimizations (R1_PERF)

This document details the performance optimizations implemented to achieve >0.90x NumPy parity for I/O and data handling, and to optimize compute throughput.

## 1. Threading Model (Dynamic Scheduling)

**Problem:** Static partitioning caused stalls when one thread was slower (e.g., page fault).
**Solution:** Implemented dynamic work-stealing approximation in `ParallelFor`.
- **Mechanism:** Threads atomically claim small chunks of work (`grain_size`) from a shared index.
- **Benefit:** Load balancing is automatic. If one thread stalls, others continue claiming work.
- **File:** `include/pycauset/core/ParallelUtils.hpp`

## 2. I/O Optimization (The "Import Gap")

**Problem:** Creating large files on disk was slow due to OS zero-filling (security feature).
**Solution:**
- **Windows:** Used `SetFileValidData` to extend file size without zero-filling. Requires `SE_MANAGE_VOLUME_NAME` privilege (enabled automatically).
- **Linux:** Used `fallocate` to pre-allocate blocks.
- **Prefetching:** Used `PrefetchVirtualMemory` (Windows) and `MAP_POPULATE` (Linux) to pre-populate page tables.
- **File:** `src/core/MemoryMapper.cpp`, `src/core/PersistentObject.cpp`

## 3. AVX-512 Optimizations

**Problem:** Bit-matrix operations were not utilizing modern CPU instructions.
**Solution:**
- **Alignment:** `DenseBitMatrix` strides are now aligned to 64 bytes (512 bits).
- **Intrinsics:** Implemented `_mm512_popcnt_epi64` and `_mm512_and_si512` for bit-matrix multiplication.
- **Runtime Dispatch:** `CpuSolver` checks for AVX-512 support at runtime using `__cpuid`.
- **File:** `src/matrix/DenseBitMatrix.cpp`, `src/compute/cpu/CpuSolver.cpp`

## 4. Memory Governor & Direct Path ("Anti-Nanny")

**Problem:** The "Streaming Solver" (out-of-core) has overhead. For datasets that fit in RAM, this overhead is unnecessary.
**Solution:**
- **Direct Path:** If `total_operation_bytes < available_ram`, the solver bypasses the streaming logic and calls BLAS/LAPACK directly.
- **Pinning:** `try_pin_memory` attempts to lock pages in RAM to prevent swapping during critical compute sections.
- **File:** `src/core/MemoryGovernor.cpp`, `src/compute/cpu/CpuSolver.cpp`

## Validation

- **Tests:** `tests/test_parallel_utils.cpp`
- **Benchmarks:** `benchmarks/benchmark_io_throughput.py`
