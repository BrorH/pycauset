# pycauset.DiagonalMatrix

```python
class pycauset.DiagonalMatrix(MatrixBase)
```

A square matrix with non-zero entries only on the diagonal.

## Construction

Use the factory, which takes a 1D vector of diagonal entries or a 2D square matrix:

```python
import pycauset as pc

D = pc.diagonal([1.0, 2.0, 3.0])        # -> pycauset.DiagonalMatrix
D = pc.diagonal([[1.0, 0.0], [0.0, 2.0]])  # extracts the diagonal
```

The raw native constructor allocates an empty matrix, filled via `set_diagonal`:

```python
D = pc.DiagonalMatrix(3)
D.set_diagonal(0, 1.0)
D.set_diagonal(1, 2.0)
D.set_diagonal(2, 3.0)
```

A dense matrix can also be marked diagonal through the properties system:

```python
D = pc.matrix([[1.0, 0.0], [0.0, 2.0]])
D.properties["is_diagonal"] = True
```

## Indexing

*   Read: `x = D[i, j]` (zero when `i != j`).
*   Write: `D[i, j] = value` (off-diagonal writes are ignored for a diagonal matrix).

## Methods

*   `set_diagonal(i, value)`: set the i-th diagonal entry.
*   `get_diagonal(i)`: read the i-th diagonal entry.

## Notes

`DiagonalMatrix` is recognized by the structure resolver as `"diagonal"`, which enables
closed-form shortcuts for `trace`, `determinant`, `matrix_rank`, and `norm`.

## See also

* [[docs/classes/matrix/pycauset.IdentityMatrix.md|pycauset.IdentityMatrix]]
* [[docs/classes/matrix/pycauset.SymmetricMatrix.md|pycauset.SymmetricMatrix]]
