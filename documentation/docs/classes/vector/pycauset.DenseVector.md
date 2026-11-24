# pycauset.DenseVector

```python
class pycauset.DenseVector(n, dtype="float64")
```

A vector of length `n` of any dtype (see the [[docs/classes/matrix/index.md|dtype table]]).
Prefer [[docs/functions/pycauset.vector.md|pycauset.vector]] (from data) or
[[docs/functions/pycauset.zeros.md|pycauset.zeros(n)]].

Element access, arithmetic, `fill`, `T`/`H` and `dtype` are described in
[[docs/classes/vector/pycauset.VectorBase.md|VectorBase]].

## Example

```python
import pycauset as pc

v = pc.DenseVector(3, dtype="int64")
v[0] = 2**62
assert v.dtype == "int64" and v[0] == 2**62
```
