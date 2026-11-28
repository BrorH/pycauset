# pycauset.invert

```python
pycauset.invert(a)    # ~a
```

NumPy's `invert`: element-wise NOT, the same function as
[[docs/functions/pycauset.bitwise_not.md|pycauset.bitwise_not]]. Logical not for dtype `bool`,
two's-complement not for integer dtypes; float and complex operands raise `TypeError`.

The matrix inverse is [[docs/functions/pycauset.inv.md|pycauset.inv]] (NumPy's `linalg.inv`).
`pycauset.invert` used to be the inverse: a NumPy name with another meaning (review 3.8 C-7).

## Example

```python
import numpy as np
import pycauset as pc

C = pc.causal_matrix(np.triu(np.ones((3, 3), bool), 1))
np.testing.assert_array_equal(np.asarray(pc.invert(C)), np.invert(np.asarray(C)))
```
