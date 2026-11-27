# pycauset.I

```python
pycauset.I(n)
pycauset.I(rows, cols)
```

The [[docs/classes/matrix/pycauset.IdentityMatrix.md|IdentityMatrix]] class under a short
name: `I(n)` is the n x n identity, `I(rows, cols)` has ones on the diagonal up to
`min(rows, cols)`. Nothing is stored. For an identity shaped like a matrix or vector, use
[[docs/functions/pycauset.identity.md|pycauset.identity]].

## Usage

```python
import pycauset as pc

eye = pc.I(1000)          # 1000 x 1000, no storage
rect = pc.I(3, 5)         # ones at (0, 0), (1, 1), (2, 2)
like = pc.identity(pc.zeros((2, 4)))   # 2 x 4, from a matrix's shape
```
