# pycauset.inv

```python
pycauset.inv(a)
```

The inverse of a square matrix, as NumPy's `linalg.inv`. (`pycauset.invert` is NumPy's
`invert`, element-wise NOT.)

Every dtype, structure and view works. The algorithm is NumPy's: LU with partial pivoting
(LAPACK `?getrf` and triangular solves on `A X = I`) in the operand's precision; integer and bool matrices invert
to float64, float32 stays float32. Structure is used where it is exact: the inverse of an
upper triangular matrix is a `TriangularMatrix`, of a diagonal matrix a `DiagonalMatrix`
(element-wise), of an identity `1/scale`; a scaled view inverts with the reciprocal scale.

## Raises

*   **ValueError**: the matrix is not square.
*   **numpy.linalg.LinAlgError** (a `ValueError`): the matrix is singular (a zero pivot),
    including every strictly upper triangular matrix such as a causal matrix, as NumPy.

## Example

```python
import pycauset as pc
import numpy as np

A = pc.matrix([[4.0, 7.0], [2.0, 6.0]])
np.testing.assert_allclose(np.asarray(pc.inv(A)), [[0.6, -0.7], [-0.2, 0.4]])
assert pc.inv(pc.matrix([[4, 7], [2, 6]])).dtype == "float64"
assert type(pc.inv(pc.diagonal([2.0, 4.0]))).__name__ == "DiagonalMatrix"
```

## See also

*   [[docs/functions/pycauset.solve.md|pycauset.solve]]
*   [[docs/functions/pycauset.pinv.md|pycauset.pinv]]
