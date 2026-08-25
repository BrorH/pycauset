# pycauset.vecdot

```python
pycauset.vecdot(a, b)
```

Returns the conjugate dot product: `sum(conj(a) * b)`.

For real inputs this equals the ordinary dot product.

## Parameters

*   **a** (vector): The first vector.
*   **b** (vector): The second vector.

## Returns

A scalar. Returns `complex` for complex inputs and `float` for real inputs.

## Examples

```python
import pycauset as pc

a = pc.vector([1+1j, 2+0j])
b = pc.vector([1-1j, 3+0j])
pc.vecdot(a, b)   # conjugate dot product
```

## See also

* [[docs/functions/pycauset.dot.md|pycauset.dot]]
* [[docs/functions/pycauset.outer.md|pycauset.outer]]
