# pycauset.eigvalsh

```python
pycauset.eigvalsh(a)
```

Eigenvalues of a real symmetric or complex Hermitian matrix, as `numpy.linalg.eigvalsh`.
Only the lower triangle is read.

## Returns

* **VectorBase**: real eigenvalues in ascending order (float32 or float64).

## Notes

Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. LAPACK `?syevd` / `?heevd`.
