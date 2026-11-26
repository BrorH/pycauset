# pycauset.cholesky

```python
pycauset.cholesky(a)
```

Cholesky factorisation of a symmetric / Hermitian positive-definite matrix, as
`numpy.linalg.cholesky`. Only the lower triangle is read.

## Returns

* **MatrixBase**: the lower-triangular factor $L$ with $L L^H = a$, dense, in the LAPACK dtype.

## Raises

* **ValueError**: the matrix is not positive definite (NumPy raises `LinAlgError`, a
  `ValueError`), or not square.

## Notes

Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. LAPACK `?potrf`.
