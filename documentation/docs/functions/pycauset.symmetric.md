# pycauset.symmetric

```python
pycauset.symmetric(data, *, rtol=None, atol=None)
```

Creates a [[docs/classes/matrix/pycauset.SymmetricMatrix.md|pycauset.SymmetricMatrix]]
($A = A^T$, packed upper triangle, about half the memory of dense) from a 2D array, in the
input's dtype (every dtype, including complex: a complex symmetric matrix is not Hermitian,
see [[docs/functions/pycauset.hermitian.md|pycauset.hermitian]]).

## Parameters

*   **data** (array-like): a 2D square array.
*   **rtol**, **atol** (float, optional): tolerances of the float/complex check (defaults `1e-5`, `1e-8`).

## Validation

Always: integer/bool input exactly, float and complex input with `numpy.allclose(..., rtol, atol)`
(the input is an array-like; only the stored triangle is kept; Python ints are int32, as in
`pycauset.matrix`). Non-square or non-conforming input raises `ValueError`; a 0 x 0 input is an
empty matrix.

## Examples

```python
import pycauset as pc

S = pc.symmetric([[2, 1], [1, 5]])   # SymmetricMatrix, int32
```

## See also

* [[docs/functions/pycauset.antisymmetric.md|pycauset.antisymmetric]]
* [[docs/functions/pycauset.hermitian.md|pycauset.hermitian]]
