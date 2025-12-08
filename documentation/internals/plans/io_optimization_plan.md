# Plan: Tiered Storage & I/O Optimization

**Status**: Draft
**Author**: GitHub Copilot
**Date**: December 8, 2025
**Target**: `v0.9.0`

## 1. Executive Summary

The current architecture treats the disk as the primary storage medium for large objects ("Disk-First"), relying on the OS to page data into RAM. While this enables infinite scaling, it introduces significant I/O latency.

This plan proposes a fundamental shift to a **Tiered Storage Architecture**.
1.  **RAM-First**: Objects exist in RAM by default. They only spill to disk when necessary.
2.  **Active Management**: A new `MemoryGovernor` actively manages RAM usage, locking hot data and evicting cold data, rather than leaving it to the OS's generic algorithms.
3.  **Proactive I/O**: When disk access is unavoidable, an `IOAccelerator` uses asynchronous prefetching and flushing to hide latency.

## 2. Architecture Overview

### New Components

*   **`MemoryGovernor` (Singleton)**: The central brain. It monitors global system RAM availability (dynamic) and manages a priority queue of all `PersistentObject` instances. It decides *who* stays in RAM and *who* goes to disk.
*   **`IOAccelerator` (Component)**: Attached to `PersistentObject`. Handles OS-specific hints (`madvise`, `PrefetchVirtualMemory`) and manages async background flushing.
*   **`StorageState` (Enum)**: New state machine for `PersistentObject`: `{ PURE_RAM, MAPPED_RAM, MAPPED_DISK }`.

### Interaction Flow

1.  **Creation**: User creates `Matrix(10GB)`.
2.  **Check**: `PersistentObject` asks `MemoryGovernor`: "Can I allocate 10GB in RAM?"
3.  **Decision**:
    *   *Case A (RAM Available)*: `MemoryGovernor` approves. Object allocates anonymous memory. **I/O Cost: 0.**
    *   *Case B (RAM Full)*: `MemoryGovernor` triggers **Eviction**. It finds the Least Recently Used (LRU) object, flushes it to disk, and unmaps it. Then it approves the new allocation.
4.  **Access**: When a solver needs to scan a matrix, it notifies `IOAccelerator`, which issues prefetch commands to the OS.

### Handling "Monstrous" Files (The 200GB Case)

For objects significantly larger than physical RAM (e.g., a 200GB matrix on a 32GB RAM system), the "All-or-Nothing" approach fails. We cannot keep the whole object in RAM.

*   **Partial Residency**: The `MemoryGovernor` and `IOAccelerator` will treat these objects as **Virtual Streams**.
*   **Sliding Window**: Instead of locking the object, we lock a **Working Set** (e.g., the 2GB chunk currently being processed).
*   **Aggressive Eviction**: As the window slides forward, the `IOAccelerator` explicitly tells the OS to discard the previous chunk (`MADV_DONTNEED`), freeing up RAM for the next chunk immediately. This prevents the OS from swapping out *other* applications (like the browser) to make room for data we are already done with.

---

## 3. Implementation Phases

### Phase 1: The Memory Governor (`core/MemoryGovernor`)

**Objective**: Create the central resource manager that prevents "Thrashing" and manages the RAM budget.

*   **Dynamic Budgeting**:
    *   The Governor will poll the OS (via `GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on Linux) to determine *actual* available RAM.
    *   It maintains a "Safety Margin" (e.g., 10% of total RAM) to allow for other apps (Chrome, OS background tasks).
*   **Priority Queue**:
    *   Implements an **ARC (Adaptive Replacement Cache)** or **LRU** policy.
    *   Objects register themselves with the Governor upon creation.
    *   `PersistentObject::touch()` method updates the object's position in the queue.
*   **Eviction Protocol**:
    *   When budget is exceeded, the Governor calls `spill_to_disk()` on the lowest-priority objects until the request can be satisfied.

**Deliverables**:
*   `include/pycauset/core/MemoryGovernor.hpp`
*   `src/core/MemoryGovernor.cpp`
*   **Tests**: Unit tests simulating low-RAM scenarios.
*   **Docs**: `internals/MemoryArchitecture.md`.

### Phase 2: Deferred Persistence (`core/PersistentObject`)

**Objective**: Stop creating files immediately. Make persistence a fallback, not a default.

*   **State Machine**:
    *   `RAM_ONLY`: Data is in `std::vector` or `malloc`. No file exists.
    *   `SYNCED`: Data is in RAM, but a file exists and is up-to-date.
    *   `DIRTY`: Data is in RAM, file exists but is stale.
    *   `DISK_BACKED`: Data is memory-mapped from disk (current behavior).
*   **Transition Logic**:
    *   `PersistentObject` starts in `RAM_ONLY`.
    *   `spill_to_disk()`: Creates the file, writes data, switches to `DISK_BACKED`.
    *   `promote_to_ram()`: Reads file into RAM, closes mapping, switches to `RAM_ONLY` (if Governor allows).
*   **The "Stall" Warning**:
    *   Transitioning from `RAM_ONLY` to `DISK_BACKED` involves writing GBs to disk. This causes a noticeable pause.
    *   **Requirement**: This behavior must be documented in `guides/Performance.md`. We will also implement a logging warning: `"PyCauset: Evicting object X to disk to free RAM (this may take a moment)..."` so the user is not confused.

**Deliverables**:
*   Refactor `PersistentObject.hpp` to support the state machine.
*   Update `MemoryMapper` to handle "Anonymous" vs "File-Backed" modes transparently.

### Phase 3: The IO Accelerator (`core/IOAccelerator`)

**Objective**: Optimize the pipe when we *do* have to use disk.

*   **Prefetching**:
    *   Implement `prefetch(offset, size)` using `PrefetchVirtualMemory` (Win) / `madvise` (Linux).
    *   Integrate into `AutoSolver`: Before a kernel runs, it calls `matrix.accelerator().prefetch_range(...)`.
*   **Sliding Window (For Huge Files)**:
    *   For files > RAM, `prefetch` implies `evict_previous`.
    *   `IOAccelerator` tracks the "Head" of the stream. When moving to chunk $N$, it prefetches $N+1$ and discards $N-1$.
*   **Async Flushing**:
    *   Implement a background thread pool for saving.
    *   `save()` returns a `std::future<void>` or similar, allowing the UI to remain responsive.

**Deliverables**:
*   `include/pycauset/core/IOAccelerator.hpp`
*   Integration into `MemoryMapper`.

### Phase 4: Lazy Copy (Object-Level CoW)

**Objective**: `B = A.copy()` should be instant and free.

*   **Mechanism**:
    *   `B` is initialized as a **Reference** to `A`'s storage.
    *   `B` registers a "Write Listener".
    *   If `B.set_element()` or `B.mutable_data()` is called:
        1.  Trigger **Deep Copy**: Allocate new storage for `B`.
        2.  Copy `A`'s data to `B`.
        3.  Deregister reference.
        4.  Perform the write.
*   **Safety**:
    *   Read-only operations (Solver) never trigger the copy.
    *   This aligns with the "Lazy Evaluation" philosophy.

---

## 4. Documentation & Protocols

Per `Protocols.md`, every step requires:

1.  **Internals**: Update `documentation/internals/MemoryArchitecture.md` (new file) explaining the Tiered Storage model.
2.  **API**: If `MemoryGovernor` exposes user settings (e.g., `set_ram_limit()`), document in `docs/classes/`.
3.  **Guides**: Update `guides/Performance.md` to explain how PyCauset manages memory and how users can tune it (if necessary).

## 5. Risk Management

*   **Risk**: `MemoryGovernor` fights with OS Paging.
    *   *Mitigation*: The Governor's limits will be conservative (soft limits). We will not use `mlock` aggressively unless explicitly requested by the user ("HPC Mode").
*   **Risk**: Data loss during Async Flush.
    *   *Mitigation*: Explicit `flush()` barrier before program exit. `PersistentObject` destructor waits for pending writes.

