# Plan: Restore "Disk-as-Memory" Mantra & Preserve Optimizations

## Core Philosophy
1.  **Infinite Canvas**: No arbitrary limits. If it doesn't fit in RAM, it goes to Disk.
2.  **Performance First**: If it fits in RAM, it stays in RAM (Pinned/Virtual) and uses optimized paths (AVX, Popcount).
3.  **Seamlessness**: The user (and the solver algorithms) should not care where the data lives.

## Current Status Audit

| Component | Storage Mechanism | Status | Issue |
|-----------|-------------------|--------|-------|
| `DenseMatrix` | `MemoryMapper` | ✅ **Good** | Automatically chooses RAM or Disk based on availability. |
| `TriangularMatrix` | `std::vector<T>` | ❌ **Critical** | **RAM Only**. Will crash on large inputs. |
| `DenseBitMatrix` | `std::vector<uint64_t>` | ❌ **Critical** | **RAM Only**. Will crash on large inputs. |
| `DiagonalMatrix` | `std::vector<T>` | ❌ **Critical** | **RAM Only**. Will crash on large inputs. |
| `CpuSolver` | Mixed | ⚠️ **Risk** | Some algorithms may allocate large temporary `std::vector` buffers. |
| `MemoryMapper` | Hybrid | ✅ **Good** | Logic exists: `if (size < available_ram) use_ram(); else use_disk();` |

## Action Plan

### Phase 1: Universal Storage Refactoring (The "Plumbing")
**Goal**: Ensure *all* matrix types use `MemoryMapper` so they inherit the "RAM if possible, Disk if necessary" behavior.

1.  **`TriangularMatrix`**:
    *   Replace `std::vector<T> data_` with `std::unique_ptr<MemoryMapper> storage_`.
    *   Update constructors to calculate size ($N(N+1)/2$) and initialize `MemoryMapper`.
    *   Ensure `get/set` methods use the mapped pointer.
    *   *Optimization Note*: Access patterns remain pointer-based, so existing loops/optimizations remain valid.

2.  **`DenseBitMatrix`**:
    *   Replace `std::vector<uint64_t> data_` with `std::unique_ptr<MemoryMapper> storage_`.
    *   Ensure bit-packing logic works on the raw `void*` from mapper.
    *   *Optimization Note*: `popcount` and bitwise ops work on raw pointers, so performance is preserved.

3.  **`DiagonalMatrix`**:
    *   Refactor to use `MemoryMapper`. (Low priority as diagonals are small $O(N)$, but necessary for consistency).

### Phase 2: Solver Safety (The "Logic")
**Goal**: Ensure algorithms don't accidentally break the mantra by allocating huge temporary buffers in RAM.

1.  **Audit `CpuSolver.cpp`**:
    *   Search for `std::vector<double>` or `new double[]` where size depends on $N$.
    *   **Fix**: Replace these with `MatrixFactory::create_temp(...)`. This ensures temporary matrices also spill to disk if needed.
    *   **Tiling**: For operations like `matmul` on huge matrices, ensure we are not trying to load the whole thing if we are just doing a block. (Current `matmul` is tiled, but we need to verify the tile buffers are small).

### Phase 3: Optimization Verification
**Goal**: Confirm that moving to `MemoryMapper` didn't kill performance for small/medium matrices.

1.  **Pointer Access**: `MemoryMapper::get_data()` returns a raw pointer.
    *   If in RAM: It's a standard pointer. Fast.
    *   If on Disk: It's a memory-mapped pointer. OS handles paging. Slower, but works.
2.  **SIMD/AVX**: These instructions work on raw pointers regardless of backing.
    *   *Check*: Ensure alignment. `MemoryMapper` should align to page boundaries (4KB), which satisfies AVX (32B/64B).

### Phase 4: Stress Testing
1.  **The "OOM" Test**:
    *   Simulate a small RAM environment (or just allocate a huge matrix > Physical RAM).
    *   Verify it runs without crashing.
2.  **The "Speed" Test**:
    *   Run existing benchmarks to ensure no regression for RAM-fitting cases.

## Execution Order
1.  Refactor `TriangularMatrix` (Most common in Causal Sets).
2.  Refactor `DenseBitMatrix` (Critical for causal relations).
3.  Audit `CpuSolver`.
4.  Run Verification.
