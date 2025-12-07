# Optimization Plan: Generalized Solver Architecture

**Date:** December 7, 2025
**Status:** Draft

## 1. Architectural Analysis

The current architecture relies on `ComputeContext` -> `AutoSolver` -> `ComputeDevice` (CPU/GPU).
*   **Dispatch**: `AutoSolver` selects a device based on matrix size.
*   **Implementation**: `CpuSolver` (and presumably `CudaSolver`) uses `dynamic_cast` chains to identify matrix types (`Dense`, `Triangular`, `Bit`, `Diagonal`) and select the appropriate kernel.

**Critique**:
*   **Patchwork Dispatch**: The `dynamic_cast` chain in `CpuSolver::matmul` is manual and error-prone. Adding a new type requires editing this monolithic function.
*   **Missing Kernels**: Many combinations (e.g., `Triangular * Dense`, `Bit * Dense`) fall back to generic, slow loops.
*   **Inversion Gaps**: `TriangularMatrix` inversion is missing entirely.
*   **Vector Ops**: `VectorOperations.cpp` is standalone and doesn't use `ComputeContext`, missing out on potential GPU acceleration or unified threading logic.

## 2. Strategy: Generalized Dispatch

To avoid "patchwork", we will implement a **Double Dispatch** mechanism or a **Registry-based Dispatch** within `CpuSolver`.

### Proposed Structure: `KernelRegistry`
Instead of hardcoded `if/else` chains, `CpuSolver` will use a registry of kernels.
*   **Key**: `(MatrixType A, MatrixType B, Operation Op)`
*   **Value**: Function Pointer / Functor to the optimized kernel.

However, given the limited number of types (Dense, Triangular, Diagonal, Bit), a **Templated Visitor** or **Static Dispatch Table** might be simpler and faster than a dynamic map.

**Refined Strategy**:
1.  **Type Traits**: Ensure all Matrix classes expose a `MatrixType` enum (already exists).
2.  **Dispatch Table**: `CpuSolver` will use a static table or switch-case on `MatrixType` enums to route to templated implementation functions.
    *   `dispatch_matmul(A, B, C)` -> `matmul_impl<TypeA, TypeB>(A, B, C)`
3.  **Fallback**: A generic template `matmul_generic` will handle cases without specialized kernels (using iterators/accessors), but we will aim to specialize all common cases.

## 3. Specific Optimization Targets

### A. Triangular Matrix Inversion
*   **Algorithm**: Back-Substitution (for strictly triangular) or Forward-Substitution.
*   **Invertibility Check**:
    *   If `has_diagonal` is false (Strictly Triangular), it is **Nilpotent** and thus **Not Invertible** (Determinant = 0).
    *   If `has_diagonal` is true, check if any diagonal element is 0.
    *   **Metadata**: Use the `has_diagonal` flag.
*   **Implementation**: Add `inverse_triangular` kernel to `CpuSolver`.

### B. BitMatrix Operations
*   **GEMV (BitMatrix * DenseVector)**:
    *   Treat bits as 0.0/1.0.
    *   Use `std::popcount` if the vector is also integer/boolean?
    *   If vector is `double`, iterate over words. For each set bit, add the corresponding vector element.
*   **GEMM (TriangularBit * TriangularBit)**:
    *   Currently missing. Should use the same `popcount` logic as `DenseBitMatrix` but respect triangular bounds to save 50% work.

### C. Vector Operations
*   **Integration**: Move `VectorOperations.cpp` logic into `CpuSolver` (or `ComputeDevice`) to unify threading models.
*   **Optimization**:
    *   `DotProduct`: SIMD for doubles. Popcount for bits.
    *   `Add/Sub`: SIMD.

### D. Diagonal Optimization
*   **GEMV**: $O(N)$ kernel.
*   **GEMM**: $O(N^2)$ (scaling rows/cols) instead of $O(N^3)$.

## 4. Implementation Plan

### Phase 1: Vector Operation Integration
1.  Refactor `VectorOperations.cpp` to delegate to `ComputeContext::get_device()`.
2.  Add `dot`, `add_vector`, `subtract_vector` to `ComputeDevice` interface.
3.  Implement optimized CPU kernels for these.

### Phase 2: Triangular Solver
1.  Implement `inverse` for `TriangularMatrix` in `CpuSolver`.
    *   Check `has_diagonal`. Throw if false.
    *   Implement $O(N^2)$ back-substitution.
2.  Implement `matmul` for `Triangular * Dense` and `Dense * Triangular`.

### Phase 3: BitMatrix GEMV
1.  Implement `batch_gemv` specialization for `DenseBitMatrix`.

### Phase 4: Testing & Benchmarking
1.  Create `tests/benchmark_optimization_sweep.cpp`.
2.  Test all combinations:
    *   `Triangular` inv.
    *   `Bit` * `Vector`.
    *   `Diagonal` * `Vector`.

## 5. Critical Constraints
*   **Documentation**: Update `SolverArchitecture.md` with the new dispatch logic.
*   **Interconnectedness**: Ensure `AutoSolver` correctly routes these new operations.
*   **Speed**: Verify speedups with benchmarks.
