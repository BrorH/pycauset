# pycauset.lu

```python
pycauset.lu(a)
```

LU decomposition with partial pivoting, `A = P L U` (SciPy's `scipy.linalg.lu`; NumPy has
none).

## Returns

`(P, L, U)` for an m×n input, k = min(m, n):

* **P**: m×m permutation matrix (real dtype).
* **L**: m×k unit lower-triangular.
* **U**: k×n upper-triangular.

A singular matrix still factors (U has a zero on its diagonal).

## Notes

Every dtype, structure and view, in NumPy's LAPACK dtype: integers and bool in float64,
float16/float32 in float32, complex32/complex64 in complex64. LAPACK `?getrf`.

## See also

* [[docs/functions/pycauset.qr.md|pycauset.qr]]
* [[docs/functions/pycauset.solve.md|pycauset.solve]]
