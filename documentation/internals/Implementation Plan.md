# Implementation Plan: Efficient K-Matrix Calculation

This document outlines the step-by-step plan to implement the calculation of $K = C(aI+C)^{-1}$ using the column-independent backward substitution method.

## Phase 1: Core C++ Implementation

### Step 1: Define C++ Interface
**Goal**: Establish the contract for the new functionality.
**Actions**:
*   Create `include/KMatrix.hpp`.
*   Define a standalone function `compute_k_matrix`.
*   **Signature**:
    ```cpp
    void compute_k_matrix(
        const CausalMatrix& C, 
        double a, 
        const std::string& output_path, 
        int num_threads
    );
    ```
*   Define a `TriangularFloatMatrix` class to handle the memory-mapped output of `double` values efficiently.

### Step 2: Implement the Column Solver Kernel
**Goal**: Implement the math for a single column correctly.
**Actions**:
*   Create `src/KMatrix.cpp`.
*   Implement a function `solve_column(j, C, a, &output_buffer)`:
    1.  Allocate a temporary buffer for column $j$ (size $N$).
    2.  Loop $i$ from $j-1$ down to $0$.
    3.  Compute sum $S = \sum_{m=i+1}^{j-1} C_{im} K_{mj}$.
    4.  Set $K_{ij} = (C_{ij} - S) / a$.
    5.  Write the column to the memory-mapped output.

### Step 3: Optimize the Inner Loop
**Goal**: Maximize performance by exploiting the binary nature of $C$.
**Actions**:
*   The term $C_{im}$ is a bit. We avoid floating point multiplication.
*   **Optimization**:
    *   Iterate $m$ using bit-scan operations (e.g., `std::countr_zero` or `_BitScanForward64`) on the chunks of $C$'s row $i$.
    *   Only access $K_{mj}$ indices where the bit in $C$ is set.
    *   This turns the dot product into a sparse accumulation: `if (C.test(i, m)) sum += K[m];`.

### Step 4: Parallelism and Streaming
**Goal**: Utilize multi-core CPUs and manage memory for massive $N$.
**Actions**:
*   Use OpenMP to parallelize the outer loop (over columns $j$).
*   **Chunking**: Process columns in blocks (e.g., 64 columns at a time) to improve cache locality for the rows of $C$.
*   **Memory Mapping**: Ensure the output file is grown to the correct size ($N \times N \times 8$ bytes) before starting.
*   **Safety**: Ensure threads write to distinct regions of the memory-mapped file (columns are disjoint, so this is naturally thread-safe).

## Phase 2: Python Integration

### Step 5: Python Bindings
**Goal**: Expose the C++ function to Python users.
**Actions**:
*   Modify `src/bindings.cpp`.
*   Add `compute_k` to the module definition.
*   Map the `output_path` argument to handle the file creation logic.

### Step 6: Python Wrapper
**Goal**: Provide a pythonic API.
**Actions**:
*   Update `python/pycauset/__init__.py`.
*   Add `pycauset.compute_k(matrix, a, saveas=...)`.
*   Handle default paths and temporary file cleanup if `saveas` is not provided.

## Phase 3: Verification

### Step 7: Testing
**Goal**: Ensure correctness.
**Actions**:
*   Create `tests/python/test_k_matrix.py`.
*   **Small Scale Test**:
    *   Generate a random $100 \times 100$ causal matrix $C$.
    *   Compute $K_{ref} = C @ \text{inv}(aI + C)$ using NumPy.
    *   Compute $K_{test}$ using the new C++ implementation.
    *   Assert `np.allclose(K_ref, K_test)`.
*   **Edge Cases**: $a=1$, $a=0.001$, empty matrix, full upper-triangular matrix.

### Step 8: Benchmarking
**Goal**: Verify performance gains.
**Actions**:
*   Compare execution time against a naive NumPy implementation for $N=1000, 5000$.
*   Estimate time for $N=10^6$ based on scaling.
