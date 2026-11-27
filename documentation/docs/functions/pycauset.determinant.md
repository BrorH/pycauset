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

Computed by the engine for every dtype, structure and view: complex for complex matrices, float otherwise (F-059). LU in the LAPACK dtype; structured classes use their structure (a triangular matrix multiplies its diagonal).

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ])
pc.determinant(A)   # -2.0
```

## See also

* [[docs/functions/pycauset.trace.md|pycauset.trace]]
* [[docs/functions/pycauset.slogdet.md|pycauset.slogdet]]
