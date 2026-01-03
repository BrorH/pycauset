# R1_LAZY: Lazy Evaluation & Persistence

**Status**: In Progress
**Owner**: Chief Programmer / Chief Design Engineer

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

## 3. Phased Implementation Plan

### Phase 1: The Expression Hierarchy (C++ Core)
Define the template structure that will represent lazy computations.
- [ ] **1.1 Define `MatrixExpression` Interface:** Create the base CRTP (Curiously Recurring Template Pattern) or abstract base class for expressions.
- [ ] **1.2 Implement Node Types:**
    - `ScalarExpression` (wraps a double/complex).
    - `MatrixRefExpression` (wraps a `MatrixBase` for reading).
    - `BinaryExpression` (lhs op rhs).
    - `UnaryExpression` (op child).
- [ ] **1.3 Implement Functors:** Define the operation structs (`Add`, `Sub`, `Mul`, `Sin`, `Exp`, etc.).
- [ ] **1.4 Solver Integration (Opaque Ops):**
    - Identify "Opaque" operations that cannot be fused (e.g., `MatMul`, `Inverse`).
    - Ensure these operations trigger eager evaluation via `AutoSolver` (or return a special `OpaqueExpression` that forces materialization upon access).
    - Prevent naive $O(N^3)$ loops in the expression engine.
- [ ] **1.5 DType Dispatch Strategy:**
    - Bridge the gap between runtime-polymorphic `MatrixBase` and compile-time Expression Templates.
    - Implement a dispatch mechanism (e.g., `dispatch_binary_op`) that switches on `DataType` and instantiates the correct typed Expression Template.
    - Define a type-erased wrapper or base class to hold the result.

### Phase 2: MatrixBase Integration (The "Rewrite")
Modify the existing `MatrixBase` to participate in the expression system without breaking inheritance.
- [ ] **2.1 Operator Overloading:** Change `operator+`, `operator-`, etc., in `MatrixBase` (and global scope) to return `BinaryExpression` instead of `std::unique_ptr<MatrixBase>`.
- [ ] **2.2 Assignment Evaluation:** Implement `MatrixBase::operator=(const MatrixExpression&)` to trigger the actual computation (the "Evaluation Loop").
- [ ] **2.3 Aliasing Detection:**
    - Implement `bool aliases(const MatrixBase* target)` in the expression tree.
    - In the assignment operator, check `if (expr.aliases(this))`.
    - If aliasing is detected (and unsafe, like matmul), evaluate to a temporary first, then swap/copy.
    - If safe (elementwise), evaluate in-place.
- [ ] **2.4 Property Propagation:**
    - Ensure Expressions compute their output properties (e.g., `Sym + Sym = Sym`).
    - Integrate with the `R1_PROPERTIES` system.
- [ ] **2.5 Batch Evaluation (Performance):**
    - Avoid virtual `get_element` calls per pixel.
    - Implement `fill_buffer(start, count, out_ptr)` or similar batch API in `MatrixExpression` to allow vectorized/bulk evaluation.

### Phase 3: NumPy UFunc Bridge (Python Interop)
Ensure Python users get lazy behavior even when using NumPy functions.
- [ ] **3.1 `__array_ufunc__` Hook:** Implement this method on the Python `Matrix` wrapper.
- [ ] **3.2 UFunc Mapping:** Map `np.add`, `np.sin`, etc., to the corresponding C++ Expression generators.
- [ ] **3.3 Return Policy:** Ensure these return a `LazyMatrix` (or similar wrapper) that behaves like a matrix but hasn't computed yet.

### Phase 4: Lazy Persistence (Memory Governor)
Implement the "RAM-First" policy.
- [ ] **4.1 Anonymous Memory Default:** Ensure `ObjectFactory` creates RAM-backed mappers by default.
- [ ] **4.2 Spill Trigger:** Update `MemoryGovernor` to monitor RAM usage during the "Evaluation Loop".
- [ ] **4.3 Spill Mechanism:** Implement `spill_to_disk()` in `MatrixBase` which transparently moves data from `AnonymousMemory` to `FileBackedMemory` and updates the mapper.

### Phase 5: Windows I/O Fix (Safety)
- [ ] **5.1 Implement `discard()`:** Use `VirtualUnlock` (Windows) or `MADV_DONTNEED` (Linux) to ensure "freed" RAM is actually returned to the OS.

### Phase 6: Documentation & Polish
- [ ] **6.1 File Headers:** Add technical file-level documentation to all new and modified files (`MatrixExpression.hpp`, `MatrixBase.cpp`, etc.) per the updated Documentation Protocol.
- [ ] **6.2 Internals Docs:** Update `MemoryArchitecture.md` and `Compute Architecture.md` to reflect the new Lazy Evaluation model.
- [ ] **6.3 User Guide:** Add a section to the Performance Guide explaining "Lazy Evaluation" and how to use `spill_to_disk()` if manual control is needed.

## 4. Risks & Mitigations
*   **Aliasing:** Addressed in Phase 2.3 via explicit detection.
*   **Template Bloat:** Use `MatrixBase` virtual methods for the final "get" if possible, or limit template depth.
*   **Debuggability:** Add a `to_string()` or `repr()` to Expressions so we can see the tree structure (e.g., `Add(MatrixA, Mul(MatrixB, Scalar))`).
