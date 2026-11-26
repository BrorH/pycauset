# pycauset.eig

```python
pycauset.eig(a)
```

Eigenvalues and right eigenvectors of a general square matrix, as `numpy.linalg.eig`.

## Returns

A pair `(w, v)`:

- `w`: complex eigenvalues (complex64 or complex128), in LAPACK's order.
- `v`: a complex matrix whose columns are the normalised right eigenvectors.

## Notes

- Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. LAPACK `?geev`.
- For real input, complex-conjugate eigenvalue pairs have complex-conjugate eigenvector
  columns, as in NumPy.
- For symmetric or Hermitian input prefer `pycauset.eigh` / `pycauset.eigvalsh`.
