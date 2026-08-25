# pycauset.symmetric

```python
pycauset.symmetric(data, *, rtol=None, atol=None)
```

Creates a symmetric matrix ($A = A^T$) from a 2D array or nested sequence, with validation.

## Parameters

*   **data** (array-like): A 2D square array (or nested sequence).
*   **rtol** (float, optional): Relative tolerance for the floating-point symmetry check. Defaults to `1e-5`.
*   **atol** (float, optional): Absolute tolerance for the floating-point symmetry check. Defaults to `1e-8`.

## Returns

*   Float input (`float32`/`float16`/`float64`): a [[docs/classes/matrix/pycauset.SymmetricMatrix.md|pycauset.SymmetricMatrix]] with packed upper-triangle storage (about 2x smaller than dense).
*   Integer/bool input: the corresponding dense matrix with `is_symmetric=True` asserted (exact storage, no packing in R1).

## Validation

The structure is always validated. Integer/bool input uses exact `A == A.T`; floating input uses `numpy.allclose(A, A.T, rtol=..., atol=...)`. Complex input is rejected (`TypeError`), and non-square or non-symmetric input raises `ValueError`.

## Examples

```python
import pycauset as pc

S = pc.symmetric([ [2.0, 1.0], [1.0, 5.0] ])
S.properties["is_symmetric"]   # True
```

## See also

* [[docs/functions/pycauset.antisymmetric.md|pycauset.antisymmetric]]
* [[docs/classes/matrix/pycauset.SymmetricMatrix.md|pycauset.SymmetricMatrix]]
