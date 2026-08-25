# pycauset.qr

```python
pycauset.qr(a, mode='reduced')
```

Returns the QR decomposition of a matrix: `A = Q R`.

## Parameters

*   **a** (matrix): The input matrix.
*   **mode** (str): `'reduced'` (default), `'complete'`, `'r'`, or `'raw'` (mirrors NumPy).

## Returns

A tuple `(Q, R)`, except for NumPy's `mode='r'` / `'raw'` behaviors which are forwarded.

## Notes

`mode='reduced'` uses the native LAPACK path (`dgeqrf`/`dorgqr`) when available; other modes fall back to NumPy.

## Examples

```python
import pycauset as pc

A = pc.matrix([[1.0, 2.0], [3.0, 4.0]])
Q, R = pc.qr(A)
```

## See also

* [[docs/functions/pycauset.svd.md|pycauset.svd]]
* [[docs/functions/pycauset.lu.md|pycauset.lu]]
