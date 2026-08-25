# Pycauset Philosophy & Design Principles

Pycauset is designed to handle causal sets where $N$ is large enough that $O(N^2)$ storage becomes the primary bottleneck. To achieve this, we adhere to a strict set of design principles and "mantras" that guide every architectural decision.

## Core Philosophy (North Star)

**PyCauset is _NumPy for causal sets_.**

- Users should interact with **top-level Python objects and functions** (e.g., `pycauset.matrix`, `pycauset.causal_matrix`, `pycauset.matmul`).
- We bridge the gap between abstract theory and petabyte-scale simulation without forcing physicists to become systems engineers.

## Core Mantras

### Build for Scale
* We assume every matrix *might* be 10TB, meaning we design for the worst case (disk-backed, out-of-core) first, then optimize the best case (RAM-only) second. While NumPy crashes if you allocate 200GB, PyCauset should handle it just fine.

### Be Lazy
*   Never compute what you can describe. For example, scalar multiplication (`A * 3.5`) is just a metadata update, taking $O(1)$ time and 0 bytes. 
*   Never write to disk what you can keep in RAM. For example, matrices stay in RAM until they grow too large or the user explicitly saves them. 

### Numpy Compatibility, C++ Engine
*   The API should feel like home to numpy-users and be intimately compatible. The engine is pure C++ optimized for our specific storage formats and causal set operations. Every operation aims to benchmark at  >0.90x the speed of NumPy for in-memory operations.

### Anti-Promotion (The "Smallest Type" Rule)
*   **Principle**: Data types must remain as small as possible, constantly.
*   **Implementation**: We aggressively resist type promotion.
    *   **Underpromotion**: Operations execute in the smallest selected dtype, and results are stored in that same dtype. We do *not* silently widen intermediates.
    *   **Mixed Types**: If a float participates, the result is float. Otherwise, we prefer the smallest dtype.
    *   **Overflow**: Integer *elementwise* arithmetic (add/sub/mul/div and scalar variants) follows C/NumPy two's-complement wraparound semantics: overflow wraps silently, by design. Integer *reductions* (`matmul`) use a wider internal accumulator and raise `OverflowError` on overflow; `dot` returns a Python `float` (exact). Float overflow follows IEEE-754 (`inf`/`nan`).
*   **Example**: Multiplying an `IntegerMatrix` by a float scalar (`3.5`) produces an `IntegerMatrix` with a metadata scalar factor. The data on disk remains integers.




---
