# pycauset.diagonal

```python
pycauset.diagonal(data)
```

Creates a [[docs/classes/matrix/pycauset.DiagonalMatrix.md|pycauset.DiagonalMatrix]] in the
input's dtype (every dtype) from a 1D vector of diagonal entries or the diagonal of a 2D square
matrix (off-diagonal values are ignored). A vector is written straight onto the diagonal (no
n x n array is built); a PyCauset matrix is read in place.

## Examples

```python
import pycauset as pc

D = pc.diagonal([1, 2, 3])   # DiagonalMatrix, int32 (Python ints, as in pycauset.matrix)
```

## See also

* [[docs/functions/pycauset.identity.md|pycauset.identity]]
