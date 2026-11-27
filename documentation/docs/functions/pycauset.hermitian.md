# pycauset.hermitian

```python
pycauset.hermitian(data, *, rtol=None, atol=None)
```

Creates a [[docs/classes/matrix/pycauset.HermitianMatrix.md|pycauset.HermitianMatrix]]
($A = A^H$, packed upper triangle) from a 2D array. Complex input keeps its dtype; real input
becomes complex_float64 (for a real packed type use `pycauset.symmetric`).

## Parameters

*   **data** (array-like): a 2D square array.
*   **rtol**, **atol** (float, optional): tolerances of the check `numpy.allclose(A, A.conj().T)`
    (defaults `1e-5`, `1e-8`).

## Raises

*   `ValueError`: non-square, empty or non-Hermitian input.

## See also

* [[docs/functions/pycauset.symmetric.md|pycauset.symmetric]]
