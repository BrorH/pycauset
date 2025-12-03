# pycauset.eig

```python
pycauset.eig(matrix: MatrixBase) -> tuple[ComplexVector, ComplexMatrix]
```

Compute the eigenvalues and right eigenvectors of a square matrix.

## Parameters

*   `matrix`: The input matrix.

## Returns

A tuple `(w, v)` where:
*   `w`: A `ComplexVector` containing the eigenvalues.
*   `v`: A `ComplexMatrix` where column `v[:, i]` is the eigenvector corresponding to the eigenvalue `w[i]`.

## Notes

*   **Algorithm**: Uses a parallelized **QR Algorithm** with **Hessenberg Reduction**.
    *   The matrix is first reduced to Upper Hessenberg form using Householder reflections ($O(N^3)$).
    *   Iterations are performed using implicit shifts and Givens rotations ($O(N^2)$ per step).
*   **Performance**: This approach is significantly faster than standard QR iterations, making eigenvalue calculations feasible for $N \approx 2000-5000$.
*   For Identity and Diagonal matrices, the eigenvectors are the standard basis vectors (Identity matrix).
