# pycauset.antisymmetric

```python
pycauset.antisymmetric(data, *, rtol=None, atol=None)
```

Creates an anti-symmetric (skew-symmetric) matrix ($A = -A^T$) from a 2D array or nested sequence, with validation.

## Parameters

*   **data** (array-like): A 2D square array (or nested sequence).
*   **rtol** (float, optional): Relative tolerance for the floating-point anti-symmetry check. Defaults to `1e-5`.
*   **atol** (float, optional): Absolute tolerance for the floating-point anti-symmetry check. Defaults to `1e-8`.

## Returns

*   Float input (`float32`/`float16`/`float64`): an [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]] with packed strict-upper-triangle storage.
*   Integer/bool input: the corresponding dense matrix with `is_anti_symmetric=True` asserted (exact storage, no packing in R1).

## Validation

The structure is always validated. Integer/bool input uses exact `A == -A.T`; floating input uses `numpy.allclose(A, -A.T, rtol=..., atol=...)`. The diagonal must be zero, the input must be square, and complex input is rejected (`TypeError`).

## Examples

```python
import pycauset as pc

A = pc.antisymmetric([ [0.0, 2.0], [-2.0, 0.0] ])
A.properties["is_anti_symmetric"]   # True
A.properties["has_zero_diagonal"]   # True
```

## See also

* [[docs/functions/pycauset.symmetric.md|pycauset.symmetric]]
* [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]]
