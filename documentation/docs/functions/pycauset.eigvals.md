# pycauset.eigvals

```python
pycauset.eigvals(matrix: MatrixBase) -> ComplexVector
```

Compute the eigenvalues of a general matrix.

## Parameters

*   `matrix`: The input matrix (Dense, Triangular, Diagonal, or Identity).

## Returns

*   `ComplexVector`: A vector containing the eigenvalues.

## Notes

*   For **Identity** and **Diagonal** matrices, this is an O(N) operation.
*   For **Triangular** matrices (with diagonal elements stored), the eigenvalues are simply the diagonal elements.
*   For **Dense** matrices, a QR algorithm is used to approximate the eigenvalues. This is an $O(N^3)$ operation.
*   For very large matrices ($N > 2000$), consider using [[pycauset.eigvals_arnoldi]] which computes only the top $k$ eigenvalues using an iterative method.

## See Also

*   [[pycauset.eigvals_arnoldi]]
*   [[pycauset.eig]]
