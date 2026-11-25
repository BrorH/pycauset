# Matrices

PyCauset has one class per **storage structure**. The element type is an attribute,
`A.dtype`, as in NumPy; every structure supports the dtypes listed on its page.

| class | stores | dtypes |
|---|---|---|
| [[docs/classes/matrix/pycauset.DenseMatrix.md|DenseMatrix]] | all `rows × cols` elements | all 15 |
| [[docs/classes/matrix/pycauset.TriangularMatrix.md|TriangularMatrix]] | the upper triangle (causal matrices: dtype `bool`) | all except `complex_float16` |
| [[docs/classes/matrix/pycauset.SymmetricMatrix.md|SymmetricMatrix]] | the upper triangle, `A == A.T` | all except `complex_float16` |
| [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|AntiSymmetricMatrix]] | the strict upper triangle, `A == -A.T` | signed: integers, floats, complex (not `complex_float16`) |
| [[docs/classes/matrix/pycauset.HermitianMatrix.md|HermitianMatrix]] | the upper triangle, `A == A.conj().T` | `complex_float32`, `complex_float64` |
| [[docs/classes/matrix/pycauset.SkewHermitianMatrix.md|SkewHermitianMatrix]] | the upper triangle, `A == -A.conj().T` | `complex_float32`, `complex_float64` |
| [[docs/classes/matrix/pycauset.DiagonalMatrix.md|DiagonalMatrix]] | the diagonal | all except `complex_float16` |
| [[docs/classes/matrix/pycauset.IdentityMatrix.md|IdentityMatrix]] | nothing (read-only) | all except `complex_float16` |

A complex `SymmetricMatrix` (`A == A.T`) and a `HermitianMatrix` (`A == A.conj().T`) are
different structures. Structured `complex_float16` storage is planned (review N.5.5).
A combination that does not exist raises `ValueError` saying why.

All of them share the methods of [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]].
Construct matrices with [[docs/functions/pycauset.matrix.md|pycauset.matrix]],
[[docs/functions/pycauset.zeros.md|zeros]], [[docs/functions/pycauset.causal_matrix.md|causal_matrix]]
or the class constructors.

## dtypes

| dtype | element | bytes per element |
|---|---|---|
| `bool` | bit (0/1) | 1/8 |
| `int8`, `int16`, `int32`, `int64` | signed integer | 1, 2, 4, 8 |
| `uint8`, `uint16`, `uint32`, `uint64` | unsigned integer | 1, 2, 4, 8 |
| `float16`, `float32`, `float64` | IEEE float | 2, 4, 8 |
| `complex_float16`, `complex_float32`, `complex_float64` | complex | 4, 8, 16 |

NumPy has no `complex_float16`; it exports as `complex64`.
