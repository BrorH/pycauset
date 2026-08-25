# pycauset.cross

```python
pycauset.cross(a, b)
```

Returns the 3D cross product of two length-3 vectors.

## Parameters

*   **a** (vector): A vector of length 3.
*   **b** (vector): A vector of length 3.

## Returns

A vector of length 3.

## Raises

*   `ValueError`: if either input is not of length 3.

## Examples

```python
import pycauset as pc

a = pc.vector([1.0, 0.0, 0.0])
b = pc.vector([0.0, 1.0, 0.0])
pc.cross(a, b)   # [0, 0, 1]
```

## See also

* [[docs/functions/pycauset.outer.md|pycauset.outer]]
