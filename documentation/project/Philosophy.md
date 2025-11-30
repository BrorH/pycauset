# Pycauset Philosophy & Design Principles

Pycauset is designed to handle causal sets where $N$ is large enough that $O(N^2)$ storage becomes the primary bottleneck. To achieve this, we adhere to a strict set of design principles and "mantras" that guide every architectural decision.

## Core Mantras

### 1. Disk is the Truth, RAM is a Cache
*   **Principle**: We assume datasets are larger than available RAM.
*   **Implementation**: All matrices are backed by files on disk and accessed via memory mapping (`mmap`). We never load a full matrix into heap memory unless explicitly requested for small chunks.
*   **Implication**: There is no "save" button; the state on disk is the current state.

### 2. Anti-Promotion (The "Smallest Type" Rule)
*   **Principle**: Data types must remain as small as possible, constantly.
*   **Implementation**: We aggressively resist type promotion.
    *   `BitMatrix` (1 bit/element) should not become `IntegerMatrix` (32 bits/element) unless absolutely necessary.
    *   `IntegerMatrix` should *never* silently become `FloatMatrix` (64 bits/element).
*   **Example**: Multiplying an `IntegerMatrix` by a float scalar (`3.5`) does **not** produce a `FloatMatrix`. It produces an `IntegerMatrix` with a metadata scalar factor of `3.5`. The data on disk remains integers.

### 3. Lazy Evaluation & Metadata Scaling
*   **Principle**: Don't compute what you can describe.
*   **Implementation**: Operations that can be handled by updating metadata (headers) are preferred over iterating through data.
    *   Scalar multiplication is a metadata update.
    *   Transposition (where applicable) should be a view change.
*   **Benefit**: Operations on 100GB matrices happen instantly if they are just metadata updates.

### 4. Bit-Packing is Non-Negotiable
*   **Principle**: Booleans are bits, not bytes.
*   **Implementation**: Causal matrices (adjacency matrices) are stored as packed bits. This offers an 8x storage reduction compared to `bool` or `uint8_t`. For $N=100,000$, this is the difference between 1.2GB and 10GB.

### 5. Numpy Compatibility, C++ Engine
*   **Principle**: The API should feel like home to Python data scientists.
*   **Implementation**: We mimic the `numpy` API (`shape`, `dtype`, slicing, broadcasting where possible). However, we do not use `numpy` arrays for storage. The engine is pure C++ optimized for our specific storage formats.

### 6. Explicit Persistence
*   **Principle**: Objects are persistent by default.
*   **Implementation**: When a matrix object is created, a file is created. When the Python object is garbage collected, the file *remains* (unless it was marked temporary). This allows for resuming work across sessions without serialization overhead.

### 7. Immutable-ish by Default
*   **Principle**: Operations return new objects (files), preserving history.
*   **Implementation**: `C = A + B` creates a new file for `C`. `A` and `B` are untouched. In-place modification (`A += B`) is supported but must be used consciously.

---

## Developer Ethos

> "If it fits in RAM, use Numpy. If it doesn't, use Pycauset."

When contributing to Pycauset, ask yourself:
1.  **Does this allocate heap memory proportional to N?** (If yes, reject it).
2.  **Does this operation force a read/write of the entire file?** (If yes, can it be lazy?).
3.  **Are we promoting types unnecessarily?** (If yes, use metadata).
