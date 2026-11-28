# pycauset.MatrixBase

Base class of every matrix. Not constructed directly; its methods are available on all
[[docs/classes/matrix/index.md|matrix classes]].

## Attributes

* **dtype** (*str*): element type, e.g. `"float64"`, `"int32"`, `"bool"`,
  `"complex_float32"`. It also compares equal to NumPy's dtypes and to every dtype token the
  constructors accept: `A.dtype == np.float64`, `C.dtype == "complex64"`, `== pc.float64`.
* **shape** (*tuple*): `(rows, cols)`; **size** (*int*): `rows * cols` (attributes, as in NumPy). Also `rows()`, `cols()`.
* **T**, **H**: transpose and conjugate transpose. O(1): they share storage.
* **backing_file** (*str*): `":memory:"` or the path of the file holding the data.

## Element access

* `A[i, j]`, `A.get(i, j)`: a Python `bool`, `int`, `float` or `complex`. Integers are exact.
* `A[i, j] = v`, `A.set(i, j, v)`: NumPy assignment rules. An integer that doesn't fit the
  dtype raises `OverflowError`; a float assigned to an integer dtype is truncated; writing
  outside the stored part of a structure (e.g. below the diagonal of a
  `TriangularMatrix`) raises `ValueError`.
* NumPy's indexing for every structure: `A[0]` (a row), slices, index arrays and boolean
  masks (`A[A > 5]`), for reading and assignment: `A[0:2, 1:3] = 0`,
  `A[[0, 1], [2, 3]] = [5, 6]`. Dense slices with step 1 are views, everything else a copy
  ([[guides/Matrix Guide|Matrix Guide]]).
* `A.fill(v)`: set every stored element (dense matrices).

## Linear algebra

`A @ B` and `A.trace()`, as on a NumPy array. The rest are functions, as in NumPy:
`pycauset.matmul`, `pycauset.eig`, `pycauset.eigh`, `pycauset.det`, `pycauset.slogdet`, ...

## Example

```python
import pycauset as pc

A = pc.zeros((2, 2), dtype="int16")
A[0, 1] = 300
assert A.dtype == "int16" and A[0, 1] == 300
assert (A.T)[1, 0] == 300
```
