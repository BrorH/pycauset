# pycauset.compute_k

```python
pycauset.compute_k(matrix: TriangularMatrix (bool), a: float) -> TriangularMatrix (float64)
```

Compute the matrix $K = C(aI + C)^{-1}$ for a causal matrix $C$ and scalar $a$.

This function uses an optimized column-independent backward substitution algorithm that exploits the binary and sparse nature of $C$.

## Parameters

*   **matrix** (*TriangularMatrix (bool)*): The input causal matrix $C$.
*   **a** (*float*): The scalar parameter $a$.

## Returns

*   **TriangularMatrix (float64)**: The result matrix $K$.
