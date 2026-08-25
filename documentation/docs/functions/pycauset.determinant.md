# pycauset.determinant

```python
pycauset.determinant(a)
```

Returns the determinant of a square matrix.

## Parameters

*   **a** (matrix): A square matrix.

## Returns

A scalar: the determinant.

## Notes

Structural shortcuts apply when the matrix carries a `properties` assertion:

*   `is_zero` returns `0`.
*   `is_identity` returns `1`.
*   `is_diagonal` with a known `diagonal_value` returns `diagonal_value ** n`.

Otherwise the native `determinant()` method is used, falling back to NumPy (`numpy.linalg.det`).

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ])
pc.determinant(A)   # -2.0
```

## See also

* [[docs/functions/pycauset.trace.md|pycauset.trace]]
* [[docs/functions/pycauset.slogdet.md|pycauset.slogdet]]
