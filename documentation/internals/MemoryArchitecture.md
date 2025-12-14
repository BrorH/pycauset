# Memory Architecture & Tiered Storage

**Status**: Implemented (Phases 1-4)
**Last Updated**: December 8, 2025

## Overview

PyCauset employs a **Tiered Storage Architecture** to balance the conflicting goals of "Infinite Scale" (Disk) and "High Performance" (RAM).

Instead of treating the disk as a simple extension of RAM (OS Paging), PyCauset actively manages where data lives.

### The Hierarchy

1.  **L1: Physical RAM (Hot)**
    *   Objects are stored in anonymous memory (`std::vector` or `malloc`).
    *   Fastest access.
    *   Managed by `MemoryGovernor`.
2.  **L2: Memory-Mapped Disk (Warm/Cold)**
    *   Objects are stored in `.pycauset` files and mapped into virtual memory.
    *   OS manages paging.
    *   Used when RAM is full or for persistence.

## 1. The Memory Governor (Phase 1)

The `MemoryGovernor` is a singleton that acts as the central resource manager.

### Responsibilities

1.  **Dynamic Budgeting**:
    *   It polls the OS for *actual* available RAM (`GlobalMemoryStatusEx` on Windows, `sysinfo` on Linux).
    *   It maintains a "Safety Margin" (default: 10% of RAM or 2GB) to prevent the OS from choking.
2.  **Priority Queue**:
    *   It tracks all `PersistentObject` instances in an LRU (Least Recently Used) list.
    *   When an object is accessed (`touch()`), it moves to the front of the queue.
3.  **Eviction**:
    *   When a new allocation is requested via `request_ram(size)`, the Governor checks if `Available RAM > size + margin`.
    *   If not, it attempts to evict the Least Recently Used objects (spilling them to disk) until space is created.

### Usage

```cpp
// In PersistentObject::initialize
if (MemoryGovernor::instance().request_ram(size_bytes)) {
    // Allocate in RAM
    allocate_ram_buffer();
    MemoryGovernor::instance().register_object(this, size_bytes);
} else {
    // Spill to disk immediately
    initialize_mmap_file();
}

// In Solver
MemoryGovernor::instance().touch(matrix_a);
```

## 2. IO Accelerator (Phase 3)

The `IOAccelerator` optimizes the interaction between the application and the OS paging system for L2 (Disk-backed) objects.

### Mechanism

*   **Windows**: Uses `PrefetchVirtualMemory` to issue asynchronous requests to the OS memory manager, bringing pages into RAM *before* the CPU faults on them.
*   **Linux**: Uses `madvise` with `MADV_WILLNEED` to trigger read-ahead.

### Workflow

1.  **Prefetch**: Before a heavy compute operation (e.g., matrix multiplication), the solver calls `accelerator->prefetch()`. This hints to the OS to populate the page cache.
2.  **Compute**: The CPU accesses the memory. Since pages are likely already in RAM, major page faults are minimized.
3.  **Discard**: After the operation, if the data is intermediate or unlikely to be reused soon, `accelerator->discard()` is called. This uses `OfferVirtualMemory` (Windows) or `MADV_DONTNEED` (Linux) to tell the OS these pages can be evicted immediately, freeing up RAM for other tasks.

## 4. Pinned Memory & Direct Path (Phase 4)

To achieve maximum performance for operations that fit entirely in RAM, PyCauset implements a **Direct Path** optimization that bypasses the standard paging/tiling overhead.

### The "Nanny Problem" & Anti-Nanny Logic
Standard memory-mapped files rely on the OS to page data in and out. While safe, this introduces latency (page faults).
However, forcing "Streaming/Tiling" on RAM-resident data that *could* be handled by the OS pager is also inefficient (the "Nanny Problem").

### The Solution: `should_use_direct_path()`

The `MemoryGovernor` now exposes a centralized decision method: `should_use_direct_path(total_bytes)`.

1.  **Check 1 (Pinning)**: If data fits in the **Pinned Memory Budget**, we pin it and use the BLAS Direct Path. This is the fastest possible mode.
2.  **Check 2 (Anti-Nanny)**: If data fits in **Total Available RAM** (but exceeds the pinning budget), we *still* use the Direct Path (without pinning). We trust the OS pager to handle the memory efficiently, avoiding the overhead of manual tiling.
3.  **Fallback**: Only if the data exceeds Available RAM do we switch to the **Streaming/Out-of-Core Solver**.

### Direct Path Workflow

In `CpuSolver`, before starting a heavy operation (like Matrix Multiplication):

1.  **Attempt Pinning**: Call `attempt_direct_path()`. If successful, run BLAS on pinned memory.
2.  **Check Anti-Nanny**: Call `MemoryGovernor::should_use_direct_path()`. If true, run BLAS on unpinned memory (OS Paging).
3.  **Streaming Fallback**: If both fail, run `matmul_streaming` (Manual Tiling + Prefetching).

This strategy allows PyCauset to match the performance of in-memory libraries (like NumPy) for medium-sized datasets, while gracefully falling back to the robust, tiled, out-of-core solver for massive datasets.

### Generalized Implementation

The "Direct Path" logic is implemented in a generalized helper `attempt_direct_path<T>` which supports both `double` and `float`. It is automatically attempted before falling back to the streaming/tiling logic.

### Diagram

```mermaid
graph TD
    subgraph Initial State
    A[Matrix A] -->|shared_ptr| M1[MemoryMapper 1]
    end

    subgraph After B = A.copy
    A2[Matrix A] -->|shared_ptr| M2[MemoryMapper 1]
    B[Matrix B] -->|shared_ptr| M2
    end

    subgraph After B.set_element
    A3[Matrix A] -->|shared_ptr| M2
    B2[Matrix B] -->|shared_ptr| M3[MemoryMapper 2]
    end
```
