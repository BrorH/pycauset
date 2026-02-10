# pycauset.cholesky

```python
pycauset.cholesky(a)
```

Compute the Cholesky factorization of a symmetric positive-definite matrix.

## Returns

* **MatrixBase**: Lower-triangular factor $L$ such that $L L^T = a$.

## Notes

Uses the native backend when available, otherwise falls back to NumPy.
Only real floating-point inputs are supported in the current implementation.