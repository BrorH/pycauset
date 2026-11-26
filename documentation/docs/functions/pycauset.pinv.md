# pycauset.pinv

```python
pycauset.pinv(a)
```

Moore-Penrose pseudoinverse, as `numpy.linalg.pinv` with its default cutoff.

$A = U S V^H \Rightarrow A^+ = V S^+ U^H$, where singular values at or below
$\max(m, n) \cdot \varepsilon \cdot \max(s)$ are treated as zero.

## Returns

* **MatrixBase**: an n×m dense matrix in the LAPACK dtype.

## Notes

Every dtype, structure and view. Computed in NumPy's LAPACK dtype: integers and bool in float64, float16/float32 in float32, complex32/complex64 in complex64, complex128 as is. Through `pycauset.svd` (LAPACK `?gesdd`) and the engine's matmul.

## See also

* [[docs/functions/pycauset.svd.md|pycauset.svd]]
* [[docs/functions/pycauset.lstsq.md|pycauset.lstsq]]
