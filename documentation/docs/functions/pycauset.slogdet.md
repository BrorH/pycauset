# pycauset.slogdet

```python
pycauset.slogdet(a)
```

Compute a sign/log-determinant pair.

## Returns

A tuple `(sign, logabsdet)` where:

- `sign` is `-1.0`, `0.0`, or `1.0`
- `logabsdet` is `log(abs(det(a)))` (or `-inf` if `det(a) == 0`)

## Notes

This currently uses the matrix method `a.determinant()`.
