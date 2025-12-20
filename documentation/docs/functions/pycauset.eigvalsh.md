# pycauset.eigvalsh

```python
pycauset.eigvalsh(a)
```

Eigenvalues for symmetric (or Hermitian) matrices.

## Returns

* **VectorBase**: Eigenvalues.

## Notes

Current implementation uses a NumPy fallback (`numpy.linalg.eigvalsh`).
