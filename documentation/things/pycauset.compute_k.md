```
pycauset.compute_k(matrix, a, saveas=None)
```
Compute the matrix $K = C(aI + C)^{-1}$ for a causal matrix $C$ and scalar $a$.

This function uses an optimized column-independent backward substitution algorithm that exploits the binary and sparse nature of $C$, achieving significantly better performance than standard matrix inversion.

### Parameters:
- matrix: [[pycauset.CausalMatrix]]. The input causal matrix $C$.
- a: float. The scalar parameter $a$.
- saveas: str (_optional_). Path to save the resulting matrix. If `None`, a temporary file is used.

### Returns:
[[pycauset.TriangularFloatMatrix]] instance
