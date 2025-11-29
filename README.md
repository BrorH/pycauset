# pycauset: High-Performance Causal Set Matrix Framework

## Project Overview
**Goal**: Develop a C++ framework to manipulate massive Causal Matrices ($C$) representing Causal Sets.
**Target Scale**: $N \approx 10^6$ (Hundreds of thousands to millions).
**Storage Target**: ~100GB binary data.
**Key Constraint**: Matrix $C$ is $N \times N$, Upper Triangular, Binary (0/1), with 0 on the diagonal.

## Technical Strategy

### 1. Language & Tools
*   **Language**: C++20 (for modern memory management and concepts).
*   **Build System**: CMake.
*   **Platform**: Windows (Powershell environment), but cross-platform compatible code.

### 2. Storage Architecture
To handle 100GB datasets which exceed typical RAM:
*   **Bit-Packing**: Store 64 matrix entries in a single `uint64_t` integer. This reduces space by factor of 64 compared to `bool` or `char`.
*   **Memory Mapping (mmap)**: Instead of loading the whole file into RAM, we will map the file on disk directly to the virtual address space. The OS handles paging chunks in and out of RAM transparently.
*   **Upper Triangular Optimization**: We only store the upper triangle to save ~50% space. Mapping $(i, j)$ to a linear index will be required.

### 3. Operations
*   **Bit-Parallelism**: Operations like Addition and Subtraction can be done 64-bits at a time using CPU bitwise instructions (`AND`, `OR`, `XOR`, `NOT`).
*   **Matrix Multiplication**: This is the bottleneck ($O(N^3)$). We will implement:
    *   **The Four Russians Algorithm** (Method of Four Russians) or similar bit-packed optimizations.
    *   **Block-based Multiplication**: To optimize for CPU cache and minimize disk I/O thrashing.

## Roadmap

### Phase 1: Foundation
*   Set up CMake project structure.
*   Create a basic `CausalMatrix` class.
*   Implement bit-packing logic (setting/getting bits at index $i, j$).
*   *Deliverable*: A working class for small $N$ in memory.

### Phase 2: Persistence (The "Huge" Scale)
*   Implement a `FileBackedStorage` backend.
*   Use Windows memory mapping APIs (or portable wrappers like `boost::iostreams` or raw generic implementation) to map large files.
*   *Deliverable*: Ability to create and access a 100GB matrix file without crashing RAM.

### Phase 3: Arithmetic & Logic
*   Implement `Add`, `Subtract` (XOR/Difference), `Intersection` (AND).
*   Implement Matrix Multiplication (Boolean or Integer).
*   *Deliverable*: Basic algebra on the matrix.

### Phase 4: Optimization
*   Optimize memory access patterns (tiling/blocking).
*   Parallelization (OpenMP or `std::thread`) for multi-core utilization.

## Questions for User
1.  **Algebra**: When you say "Multiply", do you mean Boolean Matrix Multiplication (where $1+1=1$) or standard arithmetic (counting paths)?
2.  **Sparsity**: Are these matrices expected to be sparse (mostly 0s) or dense?

## Developer Notes
* Build & setup instructions live in `docs/SETUP.md`.
* Python module usage lives in `docs/PYTHON.md`.
* Default Python storage lands in a hidden `.pycauset/` directory. Auto-created files are deleted at interpreter shutdown unless you set `pycauset.save = True`; explicit `backing_file` paths are never removed automatically.
* Core matrix types now inherit from `MatrixBase`, which centralizes memory-mapped storage, cleanup, and shared helpers. `CausalMatrix` (boolean, bit-packed) and `IntegerMatrix` (32-bit counts) each implement their own element access/multiplication on top of that base, mirroring how NumPy exposes multiple matrix flavors through a unified API surface.
* The Python surface now exposes the canonical lowercase `pycauset.causalmatrix` type (matching NumPy naming). `pycauset.CausalMatrix` remains as a thin alias for existing scripts.
* `pycauset.matrix(...)` is a convenience wrapper that accepts integers, nested Python lists, or NumPy arrays so you can bootstrap matrices directly from in-memory data; it now supports arbitrary dense matrices and keeps everything in RAM (use `pycauset.causalmatrix` when you need the persisted, strictly upper-triangular representation).
* Random population can now be deterministic: set `pycauset.seed = 42` (or pass `seed=` to constructors) to reuse the same pseudo-random stream across runs.
* Multiplication now mirrors NumPy: `lhs * rhs` performs elementwise intersection while `pycauset.matmul(lhs, rhs)` (or `lhs.multiply(rhs, ...)`) computes the integer-valued matrix product.
