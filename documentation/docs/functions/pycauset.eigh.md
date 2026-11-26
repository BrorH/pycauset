# pycauset.eigh

```python
pycauset.eigh(a)
```

Eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix, as
`numpy.linalg.eigh`. Only the lower triangle is read (NumPy's default `UPLO='L'`).

## Returns

A pair `(w, v)`:

- `w`: real eigenvalues in ascending order (float32 or float64).
- `v`: a matrix whose columns are the orthonormal eigenvectors, in the LAPACK dtype.

## Notes

Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. LAPACK `?syevd` / `?heevd`.

## See also

* [[docs/functions/pycauset.eigvalsh.md|pycauset.eigvalsh]]
* [[docs/functions/pycauset.eig.md|pycauset.eig]]
