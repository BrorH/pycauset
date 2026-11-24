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
* **dtype** (*str*): any real dtype (`bool`, integers, floats). Bit matrices are always
  strictly upper triangular.
* **has_diagonal** (*bool*): also store the diagonal.

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus:

* **has_diagonal** (*bool*, read-only).
* `TriangularMatrix.random(n, p=0.5, seed=None)`: random causal (`bool`) matrix.

## Example

```python
import pycauset as pc

C = pc.causal_matrix(5, seed=1)
assert type(C) is pc.TriangularMatrix and C.dtype == "bool"
paths = C @ C                     # counts of length-2 chains
assert paths.dtype == "int32"
```
