# Expression Template Engine

This directory contains the core components of the Lazy Evaluation system.

## Architecture

The system uses the **Curiously Recurring Template Pattern (CRTP)** to build a compile-time expression tree.

*   **`MatrixExpression`**: The base interface.
*   **`MatrixRefExpression`**: Wraps a runtime `MatrixBase` object.
*   **`BinaryExpression` / `UnaryExpression`**: Lazy nodes.

## Opaque Operations Policy (Phase 1.4)

**Heavy operations (Opaque Ops) are Eager.**

Operations that cannot be efficiently fused into an element-wise loop (specifically Matrix Multiplication and Inversion) are **not** lazy.

*   `A + B`: Returns `BinaryExpression` (Lazy).
*   `A * B` (MatMul): Returns `std::unique_ptr<MatrixBase>` (Eager, computed via BLAS).

**Reasoning:**
1.  **Performance:** BLAS/cuBLAS are highly optimized. Replicating them in a lazy element-wise loop would be significantly slower ($O(N^3)$ without cache blocking).
2.  **Complexity:** Detecting and optimizing nested MatMuls in an expression tree is overly complex for Release 1.

**Usage:**
When mixing eager and lazy ops:
```cpp
// A @ B is computed immediately into a temporary T.
// T + C is then a lazy expression.
D = (A * B) + C; 
```
