# pycauset.outer

```python
pycauset.outer(a, b)
```

Returns the outer product of two vectors: `out[i, j] = a[i] * b[j]`.

## Parameters

*   **a** (vector): The first vector.
*   **b** (vector): The second vector.

## Returns

A matrix whose shape is `(len(a), len(b))`.

## Examples

```python
import pycauset as pc

u = pc.vector([1.0, 2.0])
v = pc.vector([3.0, 4.0])
pc.outer(u, v)   # [ [3, 4], [6, 8] ]
```

## See also

* [[docs/functions/pycauset.vecdot.md|pycauset.vecdot]]
