# R1_LAZY: Lazy Evaluation & Persistence

## 1. The Problem
Currently, PyCauset uses **Eager Evaluation**.
*   `C = A + B` allocates memory for `C` and computes it immediately.
*   `D = A + B + C` allocates a temporary `T = A + B`, then allocates `D`, then computes `D = T + C`.
*   **Result:** Double memory usage, double bandwidth usage.

Also, PyCauset uses **Eager Persistence** (sometimes).
*   We need a strict "RAM-First" policy where files are only created when necessary.

## 2. The Solution: Expression Templates
We will rewrite `MatrixBase` to return lightweight `Expression` objects instead of materialized matrices.

### 2.1 The Expression Hierarchy
```cpp
template <typename L, typename R, typename Op>
class BinaryExpression : public MatrixExpression {
    const L& lhs_;
    const R& rhs_;
    // ...
    double get(i, j) const { return Op::apply(lhs_.get(i, j), rhs_.get(i, j)); }
};
```

### 2.2 Lazy Persistence (The "Spill" Policy)
*   **Default:** New matrices are `AnonymousMemory` (RAM).
*   **Trigger:** `MemoryGovernor` checks RAM usage. If > Limit, it triggers a **Spill**.
*   **Spill:** Convert `AnonymousMemory` to `FileBackedMemory` (`.tmp` file) transparently.

## 3. Deliverables
*   [ ] **Expression Template Engine:**
    *   `MatrixExpression` base class.
    *   `BinaryExpression`, `UnaryExpression`, `ScalarExpression`.
    *   `MatrixBase` operator overloads returning Expressions.
    *   Assignment operator `MatrixBase::operator=(Expression)` triggers evaluation.
*   [ ] **NumPy UFunc Bridge:**
    *   Implement `__array_ufunc__` in Python `Matrix` wrapper.
    *   Capture `np.sin`, `np.exp`, `np.add`, etc.
    *   Return `Expression` objects (lazy) instead of `np.ndarray` (eager/dense).
*   [ ] **Lazy Persistence Manager:**
    *   `MemoryGovernor` integration.
    *   `spill_to_disk()` method in `MatrixBase`.
*   [ ] **Windows I/O Fix:**
    *   Implement `discard()` using `VirtualUnlock` or `OfferVirtualMemory` to ensure "freed" RAM is actually returned to the OS.

## 4. Risks
*   **Complexity:** Debugging template errors is hard.
*   **Aliasing:** `A = A + B` requires care (read/write dependency).
