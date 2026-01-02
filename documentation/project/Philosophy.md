# Pycauset Philosophy & Design Principles

Pycauset is designed to handle causal sets where $N$ is large enough that $O(N^2)$ storage becomes the primary bottleneck. To achieve this, we adhere to a strict set of design principles and "mantras" that guide every architectural decision.

## Core Philosophy (North Star)

**PyCauset is _NumPy for causal sets_.**

- Users should interact with **top-level Python objects and functions** (e.g., `pycauset.matrix`, `pycauset.causal_matrix`, `pycauset.matmul`).
- We bridge the gap between abstract theory and petabyte-scale simulation without forcing physicists to become systems engineers.

## Developer Ethos

> "We don't write 'Happy Path' code. We write code that survives a power outage, a full disk, and a 3-week runtime."

Originally, PyCauset was built solely for scale ("If it fits in RAM, use numpy"). However, as the project has evolved into a full-fledged framework for Causal Set Theory, it now offers unique value even for small simulations:
*   **Spacetime Generation**: Built-in manifolds and sprinkling algorithms.
*   **Causal Set Abstractions**: Objects that bundle topology, geometry, and causality.
*   **Reproducibility**: Standardized serialization formats (`.pycauset`).

## Core Mantras

### 1. Scale First, Ask Questions Later
*   **Principle**: We assume every matrix *might* be 10TB.
*   **Implementation**: We design for the worst case (disk-backed, out-of-core) first, then optimize the best case (RAM-only) second.
*   **Differentiation**: NumPy crashes if you allocate 100GB. PyCauset just shrugs and opens a file.

### 2. Numpy Compatibility, C++ Engine
*   **Principle**: The API should feel like home to numpy-users and be intimately compatible.
*   **Implementation**: We mimic the `numpy` API (`shape`, `dtype`, slicing, broadcasting where possible). However, we do not use `numpy` arrays for storage. The engine is pure C++ optimized for our specific storage formats and causal set operations.

### 3. Lazy is Smart
*   **Principle**: Never compute what you can describe. Never write to disk what you can keep in RAM.
*   **Implementation**:
    *   **Lazy Persistence**: Matrices stay in RAM until they grow too large or the user explicitly saves them.
    *   **Expression Templates**: Operations like `C = A * B + D` are fused and computed on-the-fly, avoiding temporary files.
    *   **Metadata Scaling**: Scalar multiplication (`A * 3.5`) is just a metadata update, taking $O(1)$ time and 0 bytes.

### 4. Properties are Gospel
*   **Principle**: If a matrix is tagged `is_diagonal=True`, we *never* check the zeros. We trust the tag.
*   **Implementation**: We use metadata to short-circuit expensive computations ("Fly Swatting"). If a property makes an operation trivial (or impossible), we return the result instantly without touching the data.

### 5. The Hardware is a Team
*   **Principle**: The CPU, GPU, and Disk are coworkers, not rivals.
*   **Implementation**: We don't just "switch" to GPU. We use **Cooperative Computing**: the CPU prefetches data from disk while the GPU crunches the previous block. We use the right tool for the job (CPU for complex logic, GPU for brute force, Disk for infinite capacity).

### 6. Anti-Promotion (The "Smallest Type" Rule)
*   **Principle**: Data types must remain as small as possible, constantly.
*   **Implementation**: We aggressively resist type promotion.
    *   **Underpromotion**: Operations execute in the smallest selected dtype, and results are stored in that same dtype. We do *not* silently widen intermediates.
    *   **Mixed Types**: If a float participates, the result is float. Otherwise, we prefer the smallest dtype.
    *   **Overflow**: Integer overflow is a hard error (no auto-promotion). Float overflow follows IEEE-754 (`inf`/`nan`).
*   **Example**: Multiplying an `IntegerMatrix` by a float scalar (`3.5`) produces an `IntegerMatrix` with a metadata scalar factor. The data on disk remains integers.

### 7. Efficient Storage & Persistence
*   **Principle**: Storage should be minimized and persistent.
*   **Implementation**:
    *   **Bit-Packing**: Causal matrices are stored as packed bits, offering an 8x storage reduction compared to `bool` or `uint8_t`.
    *   **Persistence**: Memory-mapped matrices are persistent by nature. RAM-based matrices are transient but can be easily converted to disk-backed ones to "save" a computation.


---
