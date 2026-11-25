# pycauset.eig

```python
pycauset.eig(a)
```

Eigen-decomposition for general (non-symmetric) square matrices.

## Returns

A pair `(w, v)` where:

- `w` is a complex vector of eigenvalues.
- `v` is a complex matrix whose columns are the corresponding right eigenvectors.

## Backend

- Native: LAPACK `geev` on the CPU.
- Falls back to NumPy (`numpy.linalg.eig`) if the native backend cannot handle
  the input.

## Notes

- Complex conjugate eigenvalue pairs share a pair of eigenvector columns,
  matching LAPACK/NumPy conventions.
- For symmetric/Hermitian inputs prefer `pycauset.eigh` / `pycauset.eigvalsh`.
