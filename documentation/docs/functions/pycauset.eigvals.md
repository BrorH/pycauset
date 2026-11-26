# pycauset.eigvals

```python
pycauset.eigvals(a)
```

Eigenvalues of a general square matrix, as `numpy.linalg.eigvals`.

## Returns

A complex vector of eigenvalues (complex64 or complex128).

## Notes

- Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. LAPACK `?geev`.
- For symmetric or Hermitian input prefer `pycauset.eigvalsh`.
