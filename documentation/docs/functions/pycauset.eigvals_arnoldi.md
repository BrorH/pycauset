# pycauset.eigvals_arnoldi

Computes the $k$ largest magnitude eigenvalues of a matrix using Arnoldi iteration.

## Syntax

```python
evals = pycauset.eigvals_arnoldi(matrix, k, max_iter=100, tol=1e-10)
```

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `matrix` | `Matrix` | The input matrix (Dense, Triangular, etc.). |
| `k` | `int` | The number of eigenvalues to compute. |
| `max_iter` | `int` | Maximum size of the Krylov subspace (default: 100). |
| `tol` | `float` | Tolerance for convergence (default: 1e-10). |

## Returns

| Type | Description |
| :--- | :--- |
| `ComplexVector` | A vector containing the computed eigenvalues. |

## Description

`eigvals_arnoldi` uses the Arnoldi iteration method to approximate the largest magnitude eigenvalues of a matrix. This method is significantly faster than the dense QR algorithm (`eigvals`) for large matrices ($N > 1000$) and is the only feasible method for extremely large matrices ($N \ge 10^6$) where $O(N^3)$ complexity is prohibitive.

The method constructs a Krylov subspace of dimension `max_iter` and computes the eigenvalues of the projection of the matrix onto this subspace.

### Block Arnoldi Optimization
This implementation uses a **Block Arnoldi** strategy with block size $b=16$. This allows it to perform 16 Krylov steps for the cost of a single pass over the matrix data on disk. This is crucial for performance when the matrix is memory-mapped and exceeds available RAM.


## Examples

```python
import pycauset

# Create a large matrix
N = 5000
M = pycauset.FloatMatrix(N, "")
# ... fill M ...

# Compute top 10 eigenvalues
evals = pycauset.eigvals_arnoldi(M, k=10)
```

## See Also

*   [pycauset.eigvals](pycauset.eigvals.md)
*   [pycauset.eig](pycauset.eig.md)
