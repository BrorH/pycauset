# Algorithms and Math

This document details the mathematical algorithms implemented in PyCauset for linear algebra, causal set analysis, and spacetime generation.

## 1. Algorithms and Solvers

PyCauset implements a hybrid CPU/GPU strategy for eigenvalue computation and matrix operations.

### Eigenvalue Solvers

#### A. Dense Solver (Small/Medium Scale)
**Target:** $N \le 2000$ (CPU) or VRAM-limited (GPU)
**Complexity:** $O(N^3)$

*   **GPU (Preferred):** Uses NVIDIA `cuSolver` (`cusolverDn<t>geev`).
    *   Computes all eigenvalues using the QR algorithm on the GPU.
    *   Automatically handles memory workspace queries.
    *   **Fallback:** If VRAM is insufficient, automatically falls back to CPU.
*   **CPU (Fallback):** Implicit QR Algorithm.
    *   **Hessenberg Reduction:** Reduces $A$ to upper Hessenberg form $H = Q^T A Q$ using Blocked Householder updates. Parallelized.
    *   **QR Iteration:** Francis steps to converge to Schur form. Sequential.

#### B. Block Arnoldi Iteration (Massive Scale)
**Target:** $N > 2000$, up to $N=10^6+$
**Complexity:** $O(m \cdot N^2)$

*   **GPU Acceleration:** The dominant cost, Matrix-Vector Multiplication ($V = A \times Q_{block}$), is offloaded to the GPU using `cublasDgemm` / `cublasSgemm`.
*   **CPU Orchestration:** The orthogonalization (Gram-Schmidt) and basis management remain on CPU to allow the basis size to exceed GPU memory if necessary.

**Algorithm:**
1.  **Block Multiplication:** Compute $W = A \times Q_{[m:m+b]}$ (GPU Accelerated).
2.  **Block Gram-Schmidt:** Orthogonalize $b$ new vectors against basis $Q$ (CPU Parallel).
3.  **Projection:** Update Hessenberg matrix $H$.

### Matrix Inversion

*   **GPU (Preferred):** Uses `cuSolver`.
    1.  **LU Factorization:** `cusolverDn<t>getrf`. Computes $P A = L U$.
    2.  **Inversion:** `cusolverDn<t>getri`. Computes $A^{-1}$ using LU factors.
*   **CPU (Fallback):** Block Gauss-Jordan Elimination.
    *   Matrix is processed in blocks.
    *   Off-diagonal updates are fully parallelized using the custom `ThreadPool`.

### Matrix Multiplication

Matrix multiplication uses a dynamic dispatch system:

*   **GPU:** Uses `cuBLAS` (Float32/Float64) and custom kernels (BitMatrix).
    *   **Streaming Mode:** If matrices exceed VRAM, a hybrid tiling approach is used. CPU threads pack tiles of $A$ and stream them to the GPU.
*   **CPU:** 
    *   **Float/Double:** Uses OpenBLAS/MKL (if linked) or blocked multiplication parallelized via `ThreadPool`.
    *   **BitMatrix:** Uses AVX-512 optimized kernels (`_mm512_popcnt_epi64`) for bit-packed operations, achieving significant speedups over naive implementations.
    *   **Direct Path:** For RAM-resident data, the "Streaming Solver" overhead is bypassed, calling BLAS/LAPACK directly.
*   **Specialized:**
    *   **Bit Matrices:**
        *   **CPU**: Uses `std::popcount` (AVX-512/NEON hardware instruction) for ultra-fast boolean matrix multiplication and dot products. This replaces naive loops, achieving ~30x speedups.
        *   **GPU**: Uses a custom "Transpose-then-Popcount" kernel that packs 32x32 bit tiles into registers, achieving massive throughput for path counting and transitive closure.
    *   **Triangular Matrices:** Optimized block-based algorithms for inversion and multiplication.

## 2. Stateless Sprinkling (Spacetime Generation)

One of the key features of `pycauset` is its ability to handle extremely large causal sets—potentially billions of elements—without exhausting system RAM. This is achieved through a technique we call **Stateless Sprinkling**.

### The Problem

For $N = 10^9$ (1 billion) in $D=4$ dimensions, storing the coordinates alone would require ~32 GB of RAM.

### The Solution

`pycauset` avoids storing the coordinates entirely. Instead of keeping the positions of all $N$ points in memory, we only store the **seed** used to generate them.

#### Block-Based Matrix Generation

When generating the causal matrix (which is stored on disk as a memory-mapped payload inside a single-file `.pycauset` container), we process the points in blocks that fit comfortably in the CPU cache (e.g., 1024 points at a time).

1.  **Generate Block A**: We generate the coordinates for a small block of points (Row Block).
2.  **Generate Block B**: We generate the coordinates for another small block (Column Block).
3.  **Compute Sub-matrix**: We compute the causality relations between points in Block A and Block B.
4.  **Discard Coordinates**: Once the sub-matrix is computed and written to disk, the coordinates for Block A and Block B are discarded.

#### Coordinate Recovery Algorithm

Since coordinates are not stored, retrieving the position of a specific element $i$ (where $0 \le i < N$) requires re-running the generation process for that specific point. To make this efficient, we do not restart from index 0. Instead, we use a **Block-Skipping** algorithm.

1.  **Block Decomposition**: $B = \lfloor i / \text{BLOCK\_SIZE} \rfloor$, $k = i \pmod{\text{BLOCK\_SIZE} }$.
2.  **Block Seeding**: We compute a unique, deterministic seed for block $B$ using a hash of the global seed and the block index.
3.  **Fast-Forwarding**: We initialize a PRNG with `block_seed` and generate $k$ points to reach the target.

## 3. Mathematical Derivation: Retarded Propagator ($K_R$)

We need to calculate the Retarded Propagator matrix $K_R$ for massive causal sets ($N \approx 10^6$). The generalized definition for a scalar field on a causal set is:

$$ K_R = \Phi(I - b\Phi)^{-1} $$

Where:
*   $\Phi = a C$
*   $C$: $N \times N$ Causal Matrix (Strictly Upper Triangular, Binary).
*   $I$: Identity Matrix.
*   $a, b$: Scalar constants derived from the spacetime dimension $d$, sprinkling density $\rho$, and field mass $m$.

### Derivation

Direct inversion of $(I - b\Phi)$ is $O(N^3)$ and produces a dense matrix. We can transform this into a form solvable by our existing efficient kernel.

Substitute $\Phi = aC$:
$$ K_R = aC(I - abC)^{-1} $$

Let $\alpha_{eff} = -\frac{1}{ab}$. Then:
$$ K_R = -\frac{1}{b} \left[ C (\alpha_{eff}I + C)^{-1} \right] $$

The term in the brackets, $X = C(\alpha_{eff}I + C)^{-1}$, is exactly the form solved by our existing `compute_k` kernel (which solves $X = C(a_{kernel}I+C)^{-1}$).

### Why This Approach Works So Well

1.  **Computational Complexity**: $O(N^3) \to O(N^2 \cdot d)$. The term $C_{im}$ is **binary** and sparse. We only perform additions where $C_{im}=1$.
2.  **Memory Efficiency**: $O(N^2) \to O(N)$. We only need to store ONE column of $K$ in RAM at a time.
3.  **Parallelism**: Since each column $j$ is independent, we can compute all $N$ columns in parallel.
