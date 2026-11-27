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

Computed by the engine for every dtype, structure and view: an exact integer for integer matrices, float or complex otherwise (F-059).

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ])
pc.trace(A)   # 5.0
```

## See also

* [[docs/functions/pycauset.determinant.md|pycauset.determinant]]
