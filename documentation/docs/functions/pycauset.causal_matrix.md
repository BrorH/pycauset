# pycauset.causal_matrix

```python
pycauset.causal_matrix(source, populate=True, **kwargs)
```

Factory function for creating a [[docs/classes/matrix/pycauset.TriangularMatrix.md|pycauset.TriangularMatrix]] (dtype `bool`) suitable for representing a causal relation.

## Parameters

*   **source** (*int* or 2-D data): a size `n` (an n x n matrix), or square data (a PyCauset
    matrix, a NumPy array or nested lists) whose strict upper triangle becomes the relation
    (non-zero is related). Data is checked and packed in the engine; a warning names values
    other than 0 / 1 and non-zero values on or below the diagonal (which are dropped).
*   **populate** (*bool*): with a size: `True` (default) fills each relation with probability 0.5,
    `False` leaves it empty. Ignored for data.
*   **kwargs**: with a size, passed to `TriangularMatrix.random` / the constructor (e.g. `seed`).
    With data, keyword arguments raise `TypeError`.

## Returns

*   **TriangularMatrix (bool)**: A new instance of [[docs/classes/matrix/pycauset.TriangularMatrix.md|pycauset.TriangularMatrix]] (dtype `bool`).

## Example

```python
import pycauset

c = pycauset.causal_matrix(1000)
c_empty = pycauset.causal_matrix(1000, populate=False)
```
