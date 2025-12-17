# Pycauset Philosophy & Design Principles

Pycauset is designed to handle causal sets where $N$ is large enough that $O(N^2)$ storage becomes the primary bottleneck. To achieve this, we adhere to a strict set of design principles and "mantras" that guide every architectural decision.

## Core Philosophy (North Star)

**PyCauset is _NumPy for causal sets_.**

- Users should interact with **top-level Python objects and functions** (e.g., `pycauset.matrix`, `pycauset.causal_matrix`, `pycauset.matmul`).

## Developer Ethos

> "If you just need a matrix, use numpy. If you need a Causal Set, use Pycauset."

Originally, PyCauset was built solely for scale ("If it fits in RAM, use numpy"). However, as the project has evolved into a full-fledged framework for Causal Set Theory, it now offers unique value even for small simulations:
*   **Spacetime Generation**: Built-in manifolds and sprinkling algorithms.
*   **Causal Set Abstractions**: Objects that bundle topology, geometry, and causality.
*   **Reproducibility**: Standardized serialization formats (`.pycauset`).

## Core Mantras

### 1. Hybrid Storage: RAM is Fast, Disk is Infinite
*   **Principle**: Use the right tool for the job.
*   **Implementation**: Small matrices live in RAM for maximum speed, behaving exactly like NumPy arrays. Large matrices automatically spill over to disk (memory-mapped files) when they exceed a certain threshold.
*   **Implication**: The user shouldn't have to care. Whether a matrix is 1KB or 1TB, the API remains the same.

### 2. Numpy Compatibility, C++ Engine
*   **Principle**: The API should feel like home to numpy-users and be intimately compatible.
*   **Implementation**: We mimic the `numpy` API (`shape`, `dtype`, slicing, broadcasting where possible). However, we do not use `numpy` arrays for storage. The engine is pure C++ optimized for our specific storage formats.

### 3. Anti-Promotion (The "Smallest Type" Rule)
*   **Principle**: Data types must remain as small as possible, constantly.
*   **Implementation**: We aggressively resist type promotion.
    *   `BitMatrix` (1 bit/element) should not become `IntegerMatrix` (32 bits/element) unless absolutely necessary.
    *   `IntegerMatrix` should not silently become `FloatMatrix` **unless a float participates in the operation** (mixed-kind operations necessarily produce float results).
*   **Example**: Multiplying an `IntegerMatrix` by a float scalar (`3.5`) does **not** produce a `FloatMatrix`. It produces an `IntegerMatrix` with a metadata scalar factor of `3.5`. The data on disk remains integers.

#### Underpromotion definition

When PyCauset **underpromotes**, it means:

*   The operation is executed in the **smallest selected dtype**, and
*   The result is stored in that **same dtype**.

PyCauset does **not** silently widen intermediates “for accuracy” in the default path. If higher precision is required, users must explicitly request a larger dtype.

#### Promotion policy (mixed dtypes)

Mixed-dtype operations follow an explicit, centralized promotion policy.

*   Default: prefer the smallest dtype that can represent the operation’s semantics.
*   Mixed float precision: **underpromote by default** and emit a warning (configurable).
*   Bool matrices behave as numeric 0/1 matrices; operations may promote results when required (e.g. addition producing integer counts).

#### Fundamental kinds (bit / int / float)

PyCauset distinguishes three **fundamental kinds**:

*   `bit` (bit-packed boolean storage; special-case rules allowed)
*   `int` (signed/unsigned integers)
*   `float` (float32/float64)

Rules:

*   PyCauset never **promotes down** across fundamental kinds.
*   If a float participates, the result kind is float.
*   Underpromotion applies **within** a kind (e.g., `float32` vs `float64`), not across kinds.

Examples:

*   `matmul(bit, float64) -> float64`
*   `matmul(float32, float64) -> float32` (default underpromotion within float)

#### Overflow policy

Integer overflow is a hard error.

*   PyCauset does **not** automatically promote storage dtypes to avoid overflow.
*   For large integer matrix multiplication, PyCauset may emit a **risk warning** based on a conservative bound estimate; this warning is advisory and does not change the chosen dtype.

Float overflow is handled by IEEE-754 semantics (typically `inf`/`nan`). It is not “impossible”; it is simply a different failure mode than integer overflow and is usually addressed with optional validation (e.g., `isfinite` checks) rather than mandatory runtime trapping.

### 4. Lazy Evaluation & Metadata Scaling
*   **Principle**: Don't compute what you can describe.
*   **Implementation**: Operations that can be handled by updating metadata (headers) are preferred over iterating through data.
    *   Scalar multiplication is a metadata update.
    *   Transposition (where applicable) should be a view change.
*   **Benefit**: Operations on 100GB matrices happen instantly if they are just metadata updates.

### 5. Efficient Storage & Persistence
*   **Principle**: Storage should be minimized and persistent.
*   **Implementation**:
    *   **Bit-Packing**: Causal matrices are stored as packed bits, offering an 8x storage reduction compared to `bool` or `uint8_t`.
    *   **Persistence**: Memory-mapped matrices are persistent by nature. RAM-based matrices are transient but can be easily converted to disk-backed ones to "save" a computation.


---
