# pycauset.compute_k_matrix

```python
pycauset.compute_k_matrix(C: TriangularMatrix (bool), a: float, num_threads: int = 0) -> TriangularMatrix (float64)
```

Compute the matrix $K = C(aI + C)^{-1}$ for a causal matrix $C$ and scalar $a$.

This function uses an optimized column-independent backward substitution algorithm that exploits the binary and sparse nature of $C$.

## Parameters

*   **C** (*TriangularMatrix (bool)*): The input causal matrix.
*   **a** (*float*): The scalar parameter $a$.
*   **num_threads** (*int*, optional): Number of threads to use. Defaults to 0 (auto-detect).

## Returns

*   **TriangularMatrix (float64)**: The result matrix $K$.
