# pycauset.StaleThunkError

```python
class pycauset.StaleThunkError(RuntimeError)
```

Raised when a lazy block result is read after one of its inputs was written. Operations on
block matrices return lazy blocks that compute on first access, from the inputs as they were
when the operation was created (snapshot at creation). Each block pins the versions of
everything it reads:
- its leaves;
- the leaves of nested grids and cut tiles;
- the inputs of earlier block results it uses.

A write to any of them (`A[0, 0] = 1`, `set_block`) before or after the first evaluation makes
the block stale. Reading it then raises this error instead of computing from mixed values.
Recompute the operation.

## Example

```python
import pycauset as pc
import numpy as np
from pycauset._internal.blockmatrix import BlockMatrix

a = pc.matrix(np.ones((2, 2)))
out = BlockMatrix([[a]]) @ BlockMatrix([[pc.matrix(np.eye(2))]])
a[0, 0] = 5.0
try:
    out.get(0, 0)
except pc.StaleThunkError:
    out = BlockMatrix([[a]]) @ BlockMatrix([[pc.matrix(np.eye(2))]])  # recompute
```

## See also

[[internals/Block Matrices|Block Matrices]]
