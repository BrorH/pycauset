# pycauset.IdentityMatrix

```python
class pycauset.IdentityMatrix(rows, cols=None, dtype="float64")
```

The identity (ones on the diagonal), possibly rectangular. Stores no elements and is
read-only. Supported dtypes: every dtype.
[[docs/functions/pycauset.identity.md|pycauset.identity]] also accepts `[rows, cols]` or a
matrix/vector whose shape to copy.

## Example

```python
import pycauset as pc

I = pc.IdentityMatrix(2, 3)
assert I[1, 1] == 1.0 and I[1, 2] == 0.0
```
