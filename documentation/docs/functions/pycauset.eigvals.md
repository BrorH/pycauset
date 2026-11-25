# pycauset.eigvals

```python
pycauset.eigvals(a)
```

Eigenvalues of a general (non-symmetric) square matrix.

## Returns

A complex vector of eigenvalues.

## Backend

- Native: LAPACK `geev` on the CPU.
- Falls back to NumPy (`numpy.linalg.eigvals`) if the native backend cannot
  handle the input.

## Notes

- For symmetric/Hermitian inputs prefer `pycauset.eigvalsh` / `pycauset.eigh`.
