# pycauset.cross

```python
pycauset.cross(a, b)
```

Returns the 3D cross product of two length-3 vectors.

## Parameters

*   **a** (vector): A vector of length 3 in NumPy's multiply dtype (integers wrap).
*   **b** (vector): A vector of length 3 in NumPy's multiply dtype (integers wrap).

## Returns

A vector of length 3 in NumPy's multiply dtype (integers wrap).

## Raises

*   `ValueError`: if either input is not of length 3, or both are bool (NumPy's subtract rejects bool).

## Examples

```python
import pycauset as pc

a = pc.vector([1.0, 0.0, 0.0])
b = pc.vector([0.0, 1.0, 0.0])
pc.cross(a, b)   # [0, 0, 1]
```

## See also

* [[docs/functions/pycauset.outer.md|pycauset.outer]]
