# pycauset.SymmetricMatrix

```python
class pycauset.SymmetricMatrix(n, dtype="float64")
```

An `n × n` symmetric matrix (`A == A.T`) that stores only the upper triangle; writing
`A[i, j]` also sets `A[j, i]`. Supports the real dtypes. See also
[[docs/functions/pycauset.symmetric.md|pycauset.symmetric]].

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus
`SymmetricMatrix.from_triangular(T)`: the symmetric matrix whose upper triangle is that of
the `TriangularMatrix` `T`.

## Example

```python
import pycauset as pc

S = pc.SymmetricMatrix(3)
S[0, 2] = 4.0
assert S[2, 0] == 4.0
```
