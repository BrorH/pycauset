# pycauset.eigh

```python
pycauset.eigh(a)
```

Eigen-decomposition for symmetric (or Hermitian) matrices.

## Returns

A pair `(w, v)` where:

- `w` is a vector of eigenvalues
- `v` is a matrix whose columns are eigenvectors

## Notes

Current implementation uses a NumPy fallback (`numpy.linalg.eigh`).
