# pycauset.slogdet

```python
pycauset.slogdet(a)
```

Compute a sign/log-determinant pair.

## Returns

A tuple `(sign, logabsdet)` where:

- `sign` is `-1.0`, `0.0` or `1.0` for real matrices, a complex number of modulus 1 for
  complex ones
- `logabsdet` is `log(abs(det(a)))` (`-inf` when the matrix is singular)

As NumPy's `linalg.slogdet`: computed from the LU pivots, so it does not overflow where
`det` would. Also `a.slogdet()`.
