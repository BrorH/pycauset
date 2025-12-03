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

*   Currently, for dense matrices, the eigenvector computation may return identity placeholders or approximations depending on the solver implementation status.
*   For Identity and Diagonal matrices, the eigenvectors are the standard basis vectors (Identity matrix).
