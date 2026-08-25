# pycauset.trace

```python
pycauset.trace(a)
```

Returns the sum of the diagonal elements of a square matrix.

## Parameters

*   **a** (matrix): A square matrix.

## Returns

A scalar: the trace (sum of `a[i, i]`).

## Notes

Structural shortcuts apply when the matrix carries a `properties` assertion:

*   `is_zero` returns `0`.
*   `is_identity` returns `n`.
*   `is_diagonal` with a known `diagonal_value` returns `diagonal_value * n`.

Otherwise the native `trace()` method is used, falling back to NumPy.

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ])
pc.trace(A)   # 5.0
```

## See also

* [[docs/functions/pycauset.determinant.md|pycauset.determinant]]
