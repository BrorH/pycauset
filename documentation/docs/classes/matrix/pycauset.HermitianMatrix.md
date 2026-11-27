# pycauset.HermitianMatrix

```python
class pycauset.HermitianMatrix(n, dtype="complex_float64")
```

An `n × n` Hermitian matrix (`A == A.conj().T`, real diagonal) that stores only the upper
triangle; writing `A[i, j]` also sets `A[j, i] = conj(value)`. Complex dtypes
(`complex_float16`, `complex_float32`, `complex_float64`); for real dtypes a Hermitian matrix is a
[[docs/classes/matrix/pycauset.SymmetricMatrix.md|SymmetricMatrix]].

A complex `SymmetricMatrix` (`A == A.T`) is a different structure. In causal set theory,
`i` times the Pauli–Jordan operator is Hermitian.

Writing a diagonal element with a non-zero imaginary part raises `ValueError`. `A.T` is
`A.conj()` and `A.H` is `A` (both O(1) views).

## Results that stay Hermitian

`H1 + H2`, `H1 - H2` and the element-wise product `H1 * H2` of two Hermitian matrices of one
dtype are `HermitianMatrix` (computed over the stored triangle); so are `S1 ± S2` for two
`SkewHermitianMatrix`. Mixed dtypes, views and other operands give a `DenseMatrix` with the
same values.

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus
`HermitianMatrix.from_triangular(T)` (the upper triangle of a complex `TriangularMatrix`;
its diagonal must be real).

## Example

```python
import pycauset as pc

H = pc.HermitianMatrix(3, dtype="complex_float64")
H[0, 1] = 1 + 2j
assert H[1, 0] == 1 - 2j
```
