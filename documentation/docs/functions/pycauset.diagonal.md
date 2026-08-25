# pycauset.diagonal

```python
pycauset.diagonal(data)
```

Creates a diagonal matrix from a 1D vector of diagonal entries or a 2D square matrix.

## Parameters

*   **data** (array-like): A 1D vector of diagonal entries, or a 2D square matrix (whose diagonal entries are used).

## Returns

*   Float input: a [[docs/classes/matrix/pycauset.DiagonalMatrix.md|pycauset.DiagonalMatrix]].
*   Integer/bool input: the corresponding dense matrix with `is_diagonal=True` asserted.

## Examples

```python
import pycauset as pc

D = pc.diagonal([1.0, 2.0, 3.0])
D.properties["is_diagonal"]   # True
```

## See also

* [[docs/functions/pycauset.identity.md|pycauset.identity]]
* [[docs/classes/matrix/pycauset.DiagonalMatrix.md|pycauset.DiagonalMatrix]]
