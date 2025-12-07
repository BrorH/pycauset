# Hyper-Optimization Plan

This document tracks the progress of the hyper-optimization strategy for `pycauset`.

## Strategy Overview
The goal is to maximize performance by leveraging all available hardware resources (RAM, GPU) and minimizing overhead (DMA transfers, disk I/O).

## Steps

### Step 0: RAM-First Architecture
- [x] **Objective**: Utilize all available system RAM before falling back to disk-backed storage.
- [x] **Implementation**:
    - Created `SystemUtils` to detect available RAM.
    - Updated `PersistentObject` to check available RAM and use `:memory:` path if data fits.
    - Added safety reserve (10% or 500MB).
- [x] **Status**: Completed.

### Step 1: Pinned Memory (DMA Optimization)
- [x] **Objective**: Use Page-Locked (Pinned) Memory for host buffers to enable faster DMA transfers to/from GPU.
- [x] **Implementation**:
    - Updated `ComputeContext` to dispatch pinned allocation requests.
    - Updated `CudaDevice` to use `cudaHostAlloc` / `cudaFreeHost`.
    - Updated `MemoryMapper` to use pinned memory when `is_gpu_active()` is true and storage is `:memory:`.
    - Created `tests/test_pinned_memory.cpp` for validation.
- [x] **Validation**:
    - `test_pinned_memory` passed.
    - Benchmark result: **5.4x speedup** (1.6 GB/s -> 8.7 GB/s) for 100MB transfer.
    - **Extended Validation**: Verified that `TriangularMatrix`, `DenseVector`, `DenseBitMatrix`, and `DiagonalMatrix` automatically benefit from this optimization via `tests/test_pinned_memory_extended.cpp`.
- [x] **Status**: Completed.

### Step 2: General Solvers
- [ ] **Objective**: Create a unified solver interface that can dynamically switch between CPU and GPU implementations based on problem size and hardware availability.
- [ ] **Implementation**:
    - **2.1**: Create `AutoSolver` class (implements `ComputeDevice` interface).
        - Logic: `if (gpu_available && size > threshold && type_supported) use_gpu() else use_cpu()`.
    - **2.2**: Integrate `AutoSolver` into `ComputeContext`.
    - **2.3**: Refactor `DenseMatrix` to delegate operations to `ComputeContext::get_solver()`.
    - **2.4**: Refactor `TriangularMatrix`, `DiagonalMatrix`, `DenseBitMatrix` to delegate operations.
    - **2.5**: Ensure `AutoSolver` handles fallbacks gracefully (e.g., if GPU kernel missing for specific type).
- [ ] **Status**: Pending.

### Step 3: Multi-GPU Support
- [ ] **Objective**: Distribute workload across multiple GPUs.
- [ ] **Status**: Future.
