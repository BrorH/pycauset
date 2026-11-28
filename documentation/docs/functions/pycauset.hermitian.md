# pycauset.hermitian

```python
pycauset.hermitian(data, *, rtol=None, atol=None)
```

Creates a [[docs/classes/matrix/pycauset.HermitianMatrix.md|pycauset.HermitianMatrix]]
($A = A^H$, packed upper triangle) from a 2D array. Complex input keeps its dtype; real input
becomes the complex dtype of its precision (float16 / float32: complex_float32, otherwise
complex_float64; for a real packed type use `pycauset.symmetric`).

## Parameters

*   **data** (array-like): a 2D square array.
*   **rtol**, **atol** (float, optional): tolerances of the check `numpy.allclose(A, A.conj().T)`
    (defaults `1e-5`, `1e-8`).

## Raises

*   `ValueError`: non-square or non-Hermitian input (0 x 0 is an empty matrix).

## See also

* [[docs/functions/pycauset.symmetric.md|pycauset.symmetric]]
