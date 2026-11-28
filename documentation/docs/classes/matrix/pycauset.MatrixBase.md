# pycauset.MatrixBase

Base class of every matrix. Not constructed directly; its methods are available on all
[[docs/classes/matrix/index.md|matrix classes]].

## Attributes

* **dtype** (*str*): element type, e.g. `"float64"`, `"int32"`, `"bool"`.
* **shape** (*tuple*): `(rows, cols)`; **size** (*int*): `rows * cols` (attributes, as in NumPy). Also `rows()`, `cols()`.
* **T**, **H**: transpose and conjugate transpose. O(1): they share storage.
* **backing_file** (*str*): `":memory:"` or the path of the file holding the data.

## Element access

* `A[i, j]`, `A.get(i, j)`: a Python `bool`, `int`, `float` or `complex`. Integers are exact.
* `A[i, j] = v`, `A.set(i, j, v)`: NumPy assignment rules. An integer that doesn't fit the
  dtype raises `OverflowError`; a float assigned to an integer dtype is truncated; writing
  outside the stored part of a structure (e.g. below the diagonal of a
  `TriangularMatrix`) raises `ValueError`.
* Slices and index arrays work for reading dense matrices and for assignment on all
  structures: `A[0:2, 1:3] = 0`, `A[[0, 1], [2, 3]] = [5, 6]`.
* `A.fill(v)`: set every stored element (dense matrices).

## Linear algebra

`A @ B`, `A.matmul(B)`, `A.inverse()` (`invert()`), `A.eig()`, `A.eigvals()`, `A.eigh()`,
`A.eigvalsh()`, `A.trace()`, `A.determinant()`. The same operations exist as functions
(`pycauset.matmul`, `pycauset.eigh`, ...).

## Example

```python
import pycauset as pc

A = pc.zeros((2, 2), dtype="int16")
A[0, 1] = 300
assert A.dtype == "int16" and A[0, 1] == 300
assert (A.T)[1, 0] == 300
```
