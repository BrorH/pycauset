# pycauset.outer

```python
pycauset.outer(a, b)
```

Returns the outer product of two vectors: `out[i, j] = a[i] * b[j]`.

## Parameters

*   **a** (vector): The first vector.
*   **b** (vector): The second vector.

## Returns

A dense matrix of shape `(len(a), len(b))` in NumPy's multiply dtype (integers wrap; bool
is logical and). Any dtype and view of either vector; the lengths may differ. Array-likes are
flattened, as NumPy does; a PyCauset matrix raises `TypeError`.

## Examples

```python
import pycauset as pc

u = pc.vector([1.0, 2.0])
v = pc.vector([3.0, 4.0])
pc.outer(u, v)   # [ [3, 4], [6, 8] ]
```

## See also

* [[docs/functions/pycauset.vecdot.md|pycauset.vecdot]]
