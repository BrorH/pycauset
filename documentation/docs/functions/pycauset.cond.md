# pycauset.cond

```python
pycauset.cond(a, p=None)
```

Condition number with NumPy's definition.

## Parameters

* **a** (*MatrixBase*): input matrix.
* **p**: `None` or `2` (largest over smallest singular value), `-2` (the inverse ratio), or
  `1`, `-1`, `inf`, `-inf`, `'fro'`, `'nuc'` (`norm(a, p) * norm(inv(a), p)`).

## Returns

* **float**; `inf` for a singular matrix, as NumPy.
