# pycauset.AntiSymmetricMatrix

```python
class pycauset.AntiSymmetricMatrix(n, dtype="float64")
```

An `n × n` antisymmetric matrix (`A == -A.T`, zero diagonal) that stores only the strict
upper triangle; writing `A[i, j]` also sets `A[j, i] = -value`. Supports the real dtypes.
The Pauli–Jordan operator is antisymmetric; its spectrum is computed by
[[docs/functions/pycauset.eigvals_skew.md|pycauset.eigvals_skew]]. See also
[[docs/functions/pycauset.antisymmetric.md|pycauset.antisymmetric]].

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus
`AntiSymmetricMatrix.from_triangular(T)`.

## Example

```python
import pycauset as pc

A = pc.AntiSymmetricMatrix(3)
A[0, 1] = 2.0
assert A[1, 0] == -2.0 and A[1, 1] == 0.0
```
