# Cooperative Compute Architecture

**Status**: Experimental (Phase 1 Implemented)
**Last Updated**: 2025-12-08

## Overview

The Cooperative Compute Architecture (CCA) is designed to unify the CPU, GPU, and I/O subsystems of PyCauset. Instead of operating in isolation, these systems communicate via a **Lookahead Protocol**.

## The Lookahead Protocol

The core concept is **Intent-Based I/O**. Solvers (CPU/GPU) declare their memory access patterns *before* starting computation. The I/O subsystem uses these hints to optimize data placement (prefetching, caching, pinning).

### 1. Memory Hints

Defined in `include/pycauset/core/MemoryHints.hpp`.

| Pattern | Description | I/O Action |
| :--- | :--- | :--- |
| `Sequential` | Reading 0..N | Prefetch contiguous pages. |
| `Strided` | Reading columns | Scatter-gather prefetch (Windows) or batched `madvise` (Linux). |
| `Random` | Graph traversal | (Future) Load hot pages based on index. |
| `Once` | Stream processing | Prefetch + Auto-Discard. |

### 2. Component Interaction

1.  **Solver**: Analyzes the operation (e.g., Matrix Multiplication).
2.  **Solver**: Calls `matrix->hint(MemoryHint::strided(...))`.
3.  **PersistentObject**: Forwards hint to `IOAccelerator`.
4.  **IOAccelerator**: Translates hint to OS-specific syscalls (`PrefetchVirtualMemory` / `madvise`).
5.  **Solver**: Executes computation (now with fewer page faults).

## Implementation Status

### Phase 1: Core Definitions (Complete)
*   `MemoryHint` struct defined.
*   `PersistentObject::hint()` API added.
*   `IOAccelerator::process_hint()` stub added (handles `Sequential`).

### Phase 2: I/O Intelligence (Complete)
*   Implemented `Strided` support in `IOAccelerator`.
*   Added `prefetch_ranges_impl` for Windows (using `PrefetchVirtualMemory` with scatter-gather lists) and Linux (batched `madvise`).
*   Verified with unit tests.

### Phase 3: Solver Integration (Complete)
*   Updated `CpuSolver::matmul_impl` to emit hints.
    *   Detects Transposed matrices and emits `Strided` hints.
    *   Emits `Sequential` hints for standard access.
*   Verified compilation and linking.

### Phase 4: Pinned Memory (Complete)
*   Implemented "Pinning Budget" in `MemoryGovernor`.
    *   `try_pin_memory(size)`: Atomic check-and-reserve.
    *   `unpin_memory(size)`: Release budget.
    *   Default Limit: 20% of RAM or 4GB (whichever is smaller).
*   Verified with `test_memory_governor.cpp`.
