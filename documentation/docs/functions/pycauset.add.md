# pycauset.add

```python
pycauset.add(a, b)    # a + b
```

The element-wise sum with NumPy's broadcasting, as NumPy's `add`: the function form of
the `+` operator. Matrices, vectors, NumPy arrays and nested lists are accepted (array-like
operands are imported into the engine first); one operand may be a scalar. Two scalars raise
`TypeError`.

Result dtype and structure follow the operator: see
[[internals/DType System|DType System]] and [[docs/functions/pycauset.divide.md|pycauset.divide]]
for the broadcasting rules.

## Example

```python
import pycauset as pc
import numpy as np

A = pc.matrix(np.arange(6.0).reshape(2, 3))
np.testing.assert_array_equal(np.asarray(pc.add(A, np.ones(3))), np.add(np.arange(6.0).reshape(2, 3), np.ones(3)))
```
