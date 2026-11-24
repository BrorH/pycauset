# pycauset.DenseMatrix

```python
class pycauset.DenseMatrix(rows, cols=None, dtype="float64")
```

A matrix that stores every element. Supports all dtypes. Prefer
[[docs/functions/pycauset.matrix.md|pycauset.matrix]] (from data) or
[[docs/functions/pycauset.zeros.md|pycauset.zeros]] (by shape).

## Parameters

* **rows**, **cols** (*int*): shape; `cols` defaults to `rows`.
* **dtype** (*str*): see the [[docs/classes/matrix/index.md|dtype table]].

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus:

* `set_identity()`: ones on the diagonal, zeros elsewhere.
* `&`, `|`, `^`: element-wise logic for dtype `bool`.
* `DenseMatrix.random(rows, cols=None, p=0.5, seed=None)`: a `bool` matrix whose entries
  are 1 with probability `p`.

## Example

```python
import numpy as np
import pycauset as pc

A = pc.DenseMatrix(2, 3, dtype="uint8")
A[1, 2] = 255
assert np.array_equal(pc.to_numpy(A), [[0, 0, 0], [0, 0, 255]])
```
