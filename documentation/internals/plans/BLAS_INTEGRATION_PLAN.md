# OpenBLAS Integration Plan

**Goal:** Replace manual C++ compute loops with optimized BLAS (Basic Linear Algebra Subprograms) calls to achieve NumPy-level performance while retaining PyCauset's out-of-core streaming architecture.

**Strategy:** "Hybrid Engine"
- **Chassis:** PyCauset `MemoryGovernor` & `MemoryMapper` (Handles I/O, Pinning, Streaming).
- **Engine:** OpenBLAS (Handles raw compute via AVX2/AVX-512).

---

## Phase 1: Preparation & Cleanup
*Objective: Simplify the codebase before adding new dependencies.*

1.  **Retire Float16 Support** (DONE)
    *   **Reason:** OpenBLAS does not support Float16 on CPU. Custom implementation is maintenance-heavy.
    *   **Action:** Remove `Float16` dispatch logic from `CpuSolver::matmul`. (DONE)
    *   **Action:** Remove `Float16` specific unit tests. (DONE)
    *   **Action:** Mark `Float16` data type as deprecated or unsupported in documentation. (DONE)

2.  **Snapshot Benchmarks**
    *   **Action:** Record current "Native C++" performance for Float64 (already done: ~22 GFLOPS).
    *   **Action:** Record current performance for Float32 (if implemented) to have a baseline.

---

## Phase 2: Dependency Integration (Build System)
*Objective: Successfully link OpenBLAS in the Windows/MSVC environment.*

3.  **Acquire OpenBLAS** (DONE)
    *   **Action:** Configured CMake to automatically download pre-built binaries (v0.3.26) for Windows during build.

4.  **Update CMake Configuration** (DONE)
    *   **Action:** Modify `CMakeLists.txt` to find the OpenBLAS library. (DONE)
    *   **Action:** Add include directories for `cblas.h`. (DONE)
    *   **Action:** Link `pycauset_core` against `libopenblas.lib`. (DONE)
    *   **Action:** Ensure the `.dll` is copied to the build output directory so tests can run. (DONE)

5.  **Verify Linkage**
    *   **Action:** Create a minimal "Hello BLAS" C++ test file that just calls `cblas_dgemm` on a 2x2 matrix.
    *   **Action:** Compile and run this test to confirm the build system is working before touching the main code.

---

## Phase 3: Implementation (The "Surgical" Replacement)
*Objective: Swap the inner loops for BLAS calls inside the Streaming Architecture.*

6.  **Modify `CpuSolver.cpp` Headers** (DONE)
    *   **Action:** Include `<cblas.h>`. (DONE)

7.  **Implement Float64 (Double) Support** (DONE)
    *   **Action:** Update `matmul_streaming_f64`. (DONE - Refactored to template `matmul_streaming<T>`)
    *   **Change:** Inside the inner streaming loop (where we currently have `ParallelFor` loops), replace the logic with a call to `cblas_dgemm`. (DONE)
    *   **Detail:** We must calculate `LDA`, `LDB`, `LDC` (strides) correctly based on the full matrix width, even when processing a small tile. (DONE)

8.  **Implement Float32 (Single) Support** (DONE)
    *   **Action:** Create/Update `matmul_streaming_f32`. (DONE - Handled by template)
    *   **Change:** Use `cblas_sgemm`. (DONE)

9.  **Implement Complex Numbers Support**
    *   **Note:** BLAS supports "Single Complex" (`cgemm`) and "Double Complex" (`zgemm`).
    *   **Action:** Implement `matmul_streaming_c64` (Complex Double) using `cblas_zgemm`.
    *   **Action:** Implement `matmul_streaming_c32` (Complex Float) using `cblas_cgemm`.
    *   **Detail:** Ensure `std::complex<double>` memory layout matches BLAS expectations (it usually does: real, imag, real, imag...).

---

## Phase 4: Verification & Benchmarking
*Objective: Prove correctness and performance gains.*

10. **Unit Testing**
    *   **Action:** Run `test_symmetric_matrix` and other existing tests.
    *   **Action:** Create specific tests for non-square matrices to ensure `LDA`/`LDB` strides are correct (common BLAS bug).

11. **Performance Benchmarking**
    *   **Action:** Run `benchmark_native.exe` (Float64).
    *   **Target:** We expect to see GFLOPS jump from ~22 to ~200+ (matching NumPy).
    *   **Action:** Run benchmarks for Float32 and Complex types.

12. **Large Scale Test**
    *   **Action:** Run a benchmark with N > RAM_SIZE (e.g., N=30,000 on a 16GB machine).
    *   **Verify:** Ensure it does not crash (Streaming works) and runs reasonably fast (BLAS works).

---

## Phase 5: Documentation & Cleanup
*Objective: Finalize the transition.*

13. **Update Documentation**
    *   **Action:** Update `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md` with BLAS-backed status and thresholds.
    *   **Action:** Update `internals` docs to explain the BLAS dependency.
    *   **Action:** Remove old "Micro-Kernel" implementation plans.

14. **Code Cleanup**
    *   **Action:** Remove the old generic `matmul_impl` C++ loop implementation if it is no longer used by any type.

---

## Notes on Complex Numbers
*   **Float32 Complex:** Corresponds to `std::complex<float>`. BLAS routine: `cblas_cgemm`.
*   **Float64 Complex:** Corresponds to `std::complex<double>`. BLAS routine: `cblas_zgemm`.
*   **Implementation:** The streaming logic remains identical. We just pin the memory (which is $2 \times$ larger per element) and pass the pointer to the appropriate BLAS function.
