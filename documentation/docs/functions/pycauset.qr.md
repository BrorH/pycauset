# pycauset.qr

```python
pycauset.qr(a, mode='reduced')
```

QR decomposition `A = Q R`, as `numpy.linalg.qr`.

## Parameters

*   **a** (matrix): any shape.
*   **mode** (str): `'reduced'` (default; Q is m×k, R is k×n, k = min(m, n)), `'complete'`
    (Q is m×m, R is m×n) or `'r'` (R only, k×n). NumPy's `'raw'` (LAPACK's Householder
    form) is not offered: ValueError.

## Returns

`(Q, R)`, or `R` for `mode='r'`.

## Notes

Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. The same factorization as NumPy's (`?geqrf` + `?orgqr`), computed as the LQ of the transposed buffer (`?gelqf` + `?orglq`); signs match NumPy.

## Examples

```python
import pycauset as pc

A = pc.matrix([[1.0, 2.0], [3.0, 4.0]])
Q, R = pc.qr(A)
```

## See also

* [[docs/functions/pycauset.svd.md|pycauset.svd]]
* [[docs/functions/pycauset.lu.md|pycauset.lu]]
