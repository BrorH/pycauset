# pycauset.antisymmetric

```python
pycauset.antisymmetric(data, *, rtol=None, atol=None)
```

Creates an [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]]
($A = -A^T$, strict upper triangle stored, zero diagonal) from a 2D array, in the input's
dtype: signed integer, float or complex (unsigned and bool cannot negate: `ValueError`).

## Parameters

*   **data** (array-like): a 2D square array.
*   **rtol**, **atol** (float, optional): tolerances of the float/complex check (defaults `1e-5`, `1e-8`).

## Validation

Always: integer/bool input exactly, float and complex input with `numpy.allclose(..., rtol, atol)`
(the input is an array-like; only the stored triangle is kept; Python ints are int32, as in
`pycauset.matrix`). Non-square or non-conforming input raises `ValueError`; a 0 x 0 input is an
empty matrix. The diagonal must be zero.

## Examples

```python
import pycauset as pc

A = pc.antisymmetric([[0.0, 2.0], [-2.0, 0.0]])
```

## See also

* [[docs/functions/pycauset.symmetric.md|pycauset.symmetric]]
