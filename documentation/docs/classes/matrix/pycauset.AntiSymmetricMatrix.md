# pycauset.AntiSymmetricMatrix

```python
class pycauset.AntiSymmetricMatrix(SymmetricMatrix)
```

A matrix where $A_{ij} = -A_{ji}$ (skew-symmetric), with a zero diagonal.

This class stores only the strict upper triangular part of the matrix (the diagonal is structurally zero), so an $N \times N$ matrix costs about $N(N-1)/2$ stored elements.

## Construction

Use the validated factory:

```python
import pycauset as pc

A = pc.antisymmetric([ [0.0, 2.0], [-2.0, 0.0] ])   # -> pycauset.AntiSymmetricMatrix
```

The factory checks that the input satisfies $A = -A^T$ and has a zero diagonal, then stores only the strict upper triangle.

The raw native constructor `pycauset.AntiSymmetricMatrix(n)` allocates an empty $N \times N$
anti-symmetric matrix; fill the strict upper triangle with `set(i, j, value)`.

## Indexing

Element access uses NumPy-style indexing:

*   Read: `x = A[i, j]` (if `i > j`, returns the negated upper-triangle value)
*   Write: `A[i, j] = value` (if `i > j`, stores `-value` in the upper triangle)
*   Writing a non-zero value to the diagonal raises `ValueError`.

## `transpose()`

The transpose of an anti-symmetric matrix is its negation ($A^T = -A$).

## See also

* [[docs/functions/pycauset.antisymmetric.md|pycauset.antisymmetric]]
* [[docs/classes/matrix/pycauset.SymmetricMatrix.md|pycauset.SymmetricMatrix]]
