# pycauset.dot

```python
pycauset.dot(a, b)
```

Compute a dot product or matrix product, matching NumPy's `np.dot`.

Semantics:

*   vector · vector: scalar inner product (no implicit conjugation).
*   matrix · matrix: matrix multiplication.
*   matrix · vector / vector · matrix: matrix-vector product.

Matrix cases route through `pycauset.matmul`.

## Parameters

*   `a`: The first operand (vector or matrix).
*   `b`: The second operand (vector or matrix).

## Returns

*   `float | complex` for vector · vector.
*   `MatrixBase | VectorBase` for matrix cases.

## Exceptions

*   Raises `TypeError` if a vector-like `a` lacks a `.dot(...)` method.
*   Raises `ValueError` for mismatched vector sizes or matrix dimensions.

## Examples

```python
import numpy as np
import pycauset as pc

v = pc.vector([1.0, 2.0, 3.0])
assert pc.dot(v, v) == 14.0

A = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = pc.matrix(np.array([[5.0, 6.0], [7.0, 8.0]]))
C = pc.dot(A, B)  # matrix multiplication

x = pc.vector([1.0, 1.0])
assert pc.dot(A, x).shape[0] == 2  # matrix-vector product
```

## See also

*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]
*   [[docs/functions/pycauset.norm.md|pycauset.norm]]
