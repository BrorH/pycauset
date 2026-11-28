# pycauset.TriangularMatrix

```python
class pycauset.TriangularMatrix(n, dtype="float64", has_diagonal=False)
```

An `n × n` upper-triangular matrix that stores only the upper triangle: strictly above the
diagonal, or including it with `has_diagonal=True`. Reading below the stored part gives 0;
writing there raises `ValueError`. `A.T` reads as lower triangular.

A **causal matrix** is a `TriangularMatrix` of dtype `"bool"` (1 bit per element); build it
with [[docs/functions/pycauset.causal_matrix.md|pycauset.causal_matrix]].

## Parameters

* **n** (*int*): dimension.
* **dtype** (*str*): any dtype (`bool`, integers, floats, complex). A causal matrix is
  strict (`has_diagonal=False`); bit triangles with a diagonal exist too.
* **has_diagonal** (*bool*): also store the diagonal.

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus:

* **has_diagonal** (*bool*, read-only).
* `TriangularMatrix.random(n, p=0.5, seed=None)`: random causal (`bool`) matrix.

## Results of element-wise operations

Element-wise `+ - *` keep the class when the result's zero pattern allows it (`/` gives a
dense result: 0/0 is NaN):

| Operands (`U` upper triangular, as stored) | Result |
|---|---|
| `U + U`, `U - U` | `TriangularMatrix`, with a diagonal if either operand has one |
| `U ± DiagonalMatrix / IdentityMatrix` | `TriangularMatrix` with a diagonal |
| `U * X` (`X` anything but a lower triangle) | `TriangularMatrix`, with a diagonal only if both can be non-zero there |
| `U + U.T` (upper + lower) | dense (`SymmetricMatrix` when it is `A + A.T` of one matrix) |

Values and dtypes are NumPy's; only the storage is smaller. One exception, as for structured matmul: a product with a structural zero is 0 even where the
other operand holds `inf` or `nan` (NumPy gives `nan` for `0 * inf`).

## Example

```python
import pycauset as pc

C = pc.causal_matrix(5, seed=1)
assert type(C) is pc.TriangularMatrix and C.dtype == "bool"
paths = C @ C                     # counts of length-2 chains
assert paths.dtype == "int32"
```
