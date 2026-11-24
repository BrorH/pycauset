# Matrices

PyCauset has one class per **storage structure**. The element type is an attribute,
`A.dtype`, as in NumPy; every structure supports the dtypes listed on its page.

| class | stores | dtypes |
|---|---|---|
| [[docs/classes/matrix/pycauset.DenseMatrix.md|DenseMatrix]] | all `rows × cols` elements | all 15 |
| [[docs/classes/matrix/pycauset.TriangularMatrix.md|TriangularMatrix]] | the upper triangle (causal matrices: dtype `bool`) | real dtypes |
| [[docs/classes/matrix/pycauset.SymmetricMatrix.md|SymmetricMatrix]] | the upper triangle, `A == A.T` | real dtypes |
| [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|AntiSymmetricMatrix]] | the strict upper triangle, `A == -A.T` | real dtypes |
| [[docs/classes/matrix/pycauset.DiagonalMatrix.md|DiagonalMatrix]] | the diagonal | `bool`, integers, `float64` |
| [[docs/classes/matrix/pycauset.IdentityMatrix.md|IdentityMatrix]] | nothing (read-only) | `bool`, integers, `float64` |

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
