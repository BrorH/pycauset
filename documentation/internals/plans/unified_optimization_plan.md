# Unified Optimization Architecture Plan

**Status**: Draft
**Date**: 2025-12-08
**Author**: GitHub Copilot

## 1. Executive Summary

Currently, PyCauset has three powerful optimization systems that operate in isolation:
1.  **CPU**: Multi-threaded, cache-blocked solvers (`CpuSolver`).
2.  **GPU**: Massively parallel bitwise kernels (`CudaSolver`).
3.  **I/O**: Tiered storage with prefetching and discarding (`IOAccelerator`).

**The Problem**: These systems do not communicate. The CPU/GPU "demands" data, and the I/O system "reacts". This leads to pipeline stalls (CPU waiting for Disk) and redundant data movement.

**The Goal**: Implement a **"Lookahead Protocol"** (Cooperative Pipeline). Solvers will declare their **Intent** (Access Patterns) to the I/O subsystem *before* execution, allowing the `IOAccelerator` to optimize data placement proactively.

---

## 2. Architecture: The Lookahead Protocol

We will introduce a metadata exchange layer. This is a lightweight, advisory protocol.

### 2.1 Data Structures

We will define a new structure to describe memory access intent.

```cpp
// include/pycauset/core/MemoryHints.hpp

enum class AccessPattern {
    Sequential, // Reading 0..N (Standard)
    Reverse,    // Reading N..0
    Strided,    // Reading column-wise in row-major storage
    Random,     // Graph traversal / Sparse access
    Once        // Read once and discard (Stream)
};

struct MemoryHint {
    AccessPattern pattern;
    size_t start_offset;
    size_t length;
    size_t stride_bytes; // Only for Strided
    
    // Constructor helpers...
};
```

### 2.2 Component Responsibilities

#### A. The Producer (Solvers)
The `CpuSolver` and `CudaSolver` know *what* math they are about to do.
*   **Matrix Multiplication ($A \times B$)**:
    *   Matrix A is read sequentially (Row-Major). -> Send `Sequential` hint.
    *   Matrix B is read strided (Column-Major). -> Send `Strided` hint.
*   **Element-wise Ops**:
    *   Send `Sequential` hint.

#### B. The Bridge (`PersistentObject`)
`PersistentObject` (and `FloatMatrix`, `BitMatrix`) will expose a public API for hints.
*   `void hint(const MemoryHint& hint)`: Forwards the hint to the `IOAccelerator`.

#### C. The Consumer (`IOAccelerator`)
The `IOAccelerator` interprets the hint and translates it into OS-level commands.
*   **Sequential**: Issue `madvise(WILLNEED)` / `PrefetchVirtualMemory` for the range.
*   **Strided**:
    *   *Smart Logic*: If the stride is small (e.g., < 4KB), treat as Sequential (read everything).
    *   *Scatter Logic*: If stride is large, issue specific prefetch requests for the pages containing the columns (Windows `WIN32_MEMORY_RANGE_ENTRY` array is perfect for this).
*   **Once**: Issue `WILLNEED` for the immediate future, and schedule auto-`DONTNEED` (discard) behind the read head.

---

## 3. Implementation Roadmap

### Phase 1: Core Definitions & Interfaces
**Goal**: Establish the API without changing behavior.
1.  Create `include/pycauset/core/MemoryHints.hpp`.
2.  Update `PersistentObject` to accept hints.
3.  Update `IOAccelerator` to accept hints (stub implementation).

### Phase 2: I/O Intelligence (The "Brain")
**Goal**: Make `IOAccelerator` actually do something with the hints.
1.  Implement `process_hint` in `IOAccelerator`.
2.  **Windows Implementation**: Use `PrefetchVirtualMemory` with an array of ranges for Strided access.
3.  **Linux Implementation**: Loop over `madvise` calls (carefully benchmarked to avoid syscall overhead).

### Phase 3: Solver Integration (The "Voice")
**Goal**: Teach Solvers to speak the protocol.
1.  **CPU**: In `CpuSolver::matmul`, analyze the operation.
    *   If `B` is not transposed (Row-Major storage, Column-Major access), emit `Strided` hint for `B`.
2.  **GPU**: In `CudaSolver`, before `cudaMemcpy`, emit `Sequential` hint for the source memory. This ensures the copy doesn't trigger page faults.

### Phase 4: GPU-Specific Optimization (Pinned Staging)
**Goal**: Reduce `cudaMemcpy` latency.
*   *Note*: This is a stretch goal for this plan, but part of the unification.
*   If `CudaSolver` hints `Sequential` + `GPU_Transfer`, the `IOAccelerator` could try to lock the pages in RAM (`VirtualLock` / `mlock`) temporarily to prevent the OS from swapping them out during the transfer.

---

## 4. Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Overhead** | Hint calculation slows down small ops. | **Thresholding**: Only emit hints if matrix size > 10 MB. |
| **OS Limits** | Too many prefetch requests (Strided). | **Batching**: `IOAccelerator` will aggregate strided requests into chunks (e.g., 64 pages at a time). |
| **Complexity** | Solvers become cluttered with I/O logic. | **Encapsulation**: Create a helper `ScopedAccess(matrix, pattern)` RAII wrapper that sends the hint on construction. |
| **Platform** | Windows vs Linux differences. | **Abstraction**: `IOAccelerator` hides all OS calls. Solvers never see `madvise`. |

---

## 5. Testing & Validation Strategy

Following `Protocols.md`:

1.  **Unit Tests**:
    *   Test `IOAccelerator` with a mock file. Send `Strided` hint, verify correct offsets are calculated.
2.  **Integration Tests**:
    *   `benchmark_cooperative.py`: Run a large matrix multiplication (Disk-backed).
    *   Measure: Execution Time & Page Faults (via `psutil`).
    *   Expectation: `Strided` hint should reduce Page Faults compared to baseline.
3.  **Documentation**:
    *   Create `documentation/internals/CooperativeArchitecture.md`.
    *   Update `API Reference` for `PersistentObject.hint()`.

## 6. Next Steps

1.  Approve this plan.
2.  Begin Phase 1 (Core Definitions).
