# Lazy Evaluation & Expression Templates

## Overview

PyCauset implements **Lazy Evaluation** using C++ Expression Templates. This allows operations like `C = A + B + C` to be fused into a single evaluation pass, avoiding the creation of temporary matrices for intermediate results. This significantly reduces memory bandwidth usage and allocation overhead.

## Architecture

The system is built on the **Curiously Recurring Template Pattern (CRTP)**.

### Core Components

1.  **`MatrixExpression<Derived>`**: The base class for all lazy expressions. It provides the interface for:
    *   `get_element(i, j)`: Computing the value at a specific index.
    *   `rows()`, `cols()`: Dimension queries.
    *   `aliases(target)`: Checking if the expression refers to a specific matrix (for aliasing safety).
    *   `touch_operands()`: Updating the LRU status of source matrices (for memory safety).

2.  **Expression Nodes**:
    *   `MatrixRefExpression`: A lightweight wrapper around a `const MatrixBase&`. It is the leaf node of the expression tree.
    *   `ScalarExpression`: Wraps a scalar value (double).
    *   `BinaryExpression<L, R, Op>`: Represents a binary operation. Stores operands `L` and `R` by value (assuming they are lightweight expression objects) to prevent dangling references to temporary objects in chained expressions.
    *   `UnaryExpression<E, Op>`: Represents a unary operation. Stores operand `E` by value.

3.  **Functors**:
    *   Located in `pycauset/matrix/expression/Functors.hpp`.
    *   Stateless structs with a static `apply` method (e.g., `ops::Add::apply(a, b)`).

### Evaluation Model

Evaluation is triggered only when an expression is assigned to a `MatrixBase` object via `operator=`.

```cpp
// MatrixBase.hpp
template <typename E>
MatrixBase& operator=(const MatrixExpression<E>& expr) {
    // 1. Check for aliasing
    if (expr.aliases(this)) {
        throw std::runtime_error("Aliasing detected...");
    }

    // 2. Touch operands (Memory Safety)
    expr.touch_operands();

    // 3. Evaluate
    for (uint64_t i = 0; i < rows(); ++i) {
        for (uint64_t j = 0; j < cols(); ++j) {
            set_element_as_double(i, j, expr.get_element(i, j));
        }
    }
    return *this;
}
```

## Memory Safety (The "Spill" Policy)

Lazy evaluation interacts with the `MemoryGovernor` to ensure that source matrices are not evicted from RAM while they are being read.

*   **`touch_operands()`**: Before the evaluation loop begins, the assignment operator calls `touch_operands()` on the expression tree. This recursively calls `touch()` on every `MatrixBase` referenced in the tree.
*   **LRU Update**: `touch()` updates the timestamp of the matrix in the `MemoryGovernor`, moving it to the "most recently used" position. This protects it from being chosen for eviction if a spill is triggered during evaluation (e.g., if the destination matrix needs to allocate more storage).

## Windows I/O Optimization

On Windows, the `IOAccelerator` uses `VirtualUnlock` to implement `discard()`. This serves as a hint to the OS to remove pages from the working set, similar to `madvise(MADV_DONTNEED)` on Linux. This is crucial for the "RAM-First" persistence policy, allowing us to effectively free RAM without closing the file handle.

## Limitations & Future Work

1.  **Aliasing**: Currently, assignments like `A = A + B` throw an exception to prevent unsafe aliasing. Future versions should implement temporary buffering or safe in-place evaluation for elementwise ops.
2.  **Vectorization**: The current evaluation loop is scalar. Future versions should implement `fill_buffer` in the expression interface to allow SIMD-optimized batch evaluation.
3.  **Opaque Operations**: Matrix multiplication (`A * B`) is currently eager. It returns a new `MatrixBase` immediately, rather than an expression.
