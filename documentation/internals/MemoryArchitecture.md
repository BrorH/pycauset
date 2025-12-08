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

## 3. Copy-on-Write (Lazy Copy) (Phase 4)

To support efficient object duplication (e.g., `B = A.copy()`), PyCauset implements a **Copy-on-Write (CoW)** mechanism. This ensures that copying a large matrix is instant ($O(1)$), and data is only duplicated when one of the copies is modified.

### Shared Ownership

The core of this mechanism is the transition from `std::unique_ptr<MemoryMapper>` to `std::shared_ptr<MemoryMapper>` in `PersistentObject`.

*   **Shallow Copy**: When `clone()` is called, a new `PersistentObject` is created, but it points to the *same* `MemoryMapper` (and thus the same underlying file/RAM buffer).
*   **Reference Counting**: The `std::shared_ptr` tracks how many objects are using the storage.

### The `ensure_unique()` Guard

Every method that modifies data (e.g., `set()`, `set_diagonal()`) must first call `ensure_unique()`.

```cpp
void PersistentObject::ensure_unique() {
    // If we are sharing storage with others (use_count > 1)
    if (mapper_.use_count() > 1) {
        // 1. Create a new, independent storage (Deep Copy)
        auto new_mapper = mapper_->clone();
        
        // 2. Point this object to the new storage
        mapper_ = std::move(new_mapper);
        
        // Now use_count() is 1, and we are safe to write.
    }
}
```

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
