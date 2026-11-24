# pycauset.DiagonalMatrix

```python
class pycauset.DiagonalMatrix(n, dtype="float64")
```

An `n × n` diagonal matrix that stores only its `n` diagonal elements. Supported dtypes:
`bool`, the integer dtypes and `float64`. See also
[[docs/functions/pycauset.diagonal.md|pycauset.diagonal]].

## Methods

Everything in [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]], plus
`get_diagonal(i)` and `set_diagonal(i, value)`. `A.T` is `A`.

## Example

```python
import pycauset as pc

D = pc.DiagonalMatrix(3, dtype="int32")
D.set_diagonal(1, 7)
assert D[1, 1] == 7 and D[0, 1] == 0
```
