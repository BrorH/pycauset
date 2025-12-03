# pycauset.eigvals_skew

Computes the $k$ largest magnitude eigenvalues of a real skew-symmetric matrix using Block Skew-Lanczos iteration.

## Syntax

```python
evals = pycauset.eigvals_skew(matrix, k, max_iter=100, tol=1e-10)
```

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `matrix` | `Matrix` | The input matrix. Must be real skew-symmetric ($A^T = -A$). |
| `k` | `int` | The number of eigenvalues to compute. |
| `max_iter` | `int` | Maximum size of the Krylov subspace (default: 100). |
| `tol` | `float` | Tolerance for convergence (default: 1e-10). |

## Returns

| Type | Description |
| :--- | :--- |
| `ComplexVector` | A vector containing the computed eigenvalues. |

## Description

This function is specialized for **Anti-Symmetric Hermitian** (Real Skew-Symmetric) operators, which frequently appear in Quantum Field Theory on Causal Sets (e.g., the Pauli-Jordan commutator function).

### Mathematical Properties
*   **Skew-Symmetry:** $A^T = -A$.
*   **Eigenvalues:** The eigenvalues of a real skew-symmetric matrix always come in purely imaginary pairs $\pm i\lambda$ (or are zero).
*   **Algorithm:** Uses a **Block Skew-Lanczos** iteration. Unlike general Arnoldi (which requires orthogonalizing against *all* previous vectors), Skew-Lanczos only requires orthogonalizing against the previous two blocks, reducing computational complexity from $O(m^2 N)$ to $O(m N)$.

### Performance
This solver is significantly faster than `eigvals_arnoldi` for skew-symmetric matrices due to the short recurrence relation. It also preserves the pairing symmetry of the spectrum better than general solvers.

**Parallelization:**
For large matrices ($N \ge 10,000$), the solver automatically utilizes multiple threads to parallelize vector updates and dot products. You can control the number of threads using [[pycauset.set_num_threads]].
