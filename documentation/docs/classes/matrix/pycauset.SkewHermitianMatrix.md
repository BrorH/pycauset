# pycauset.SkewHermitianMatrix

```python
class pycauset.SkewHermitianMatrix(n, dtype="complex_float64")
```

An `n × n` skew-Hermitian matrix (`A == -A.conj().T`, imaginary diagonal) that stores only
the upper triangle; writing `A[i, j]` also sets `A[j, i] = -conj(value)`. Complex dtypes
(`complex_float32`, `complex_float64`); for real dtypes a skew-Hermitian matrix is an
[[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|AntiSymmetricMatrix]].

Writing a diagonal element with a non-zero real part raises `ValueError`. `A.T` is
`-A.conj()` and `A.H` is `-A` (both O(1) views).

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus
`SkewHermitianMatrix.from_triangular(T)`.

## Example

```python
import pycauset as pc

K = pc.SkewHermitianMatrix(2, dtype="complex_float64")
K[0, 1] = 2 + 1j
assert K[1, 0] == -2 + 1j
```
