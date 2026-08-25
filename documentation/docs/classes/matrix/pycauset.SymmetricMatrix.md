# pycauset.SymmetricMatrix

```python
class pycauset.SymmetricMatrix(MatrixBase)
```

A matrix where $A_{ij} = A_{ji}$.

This class stores only the upper triangular part of the matrix (including the diagonal) to save memory, so an $N \times N$ matrix costs about $N(N+1)/2$ elements instead of $N^2$.

## Construction

Use the validated factory:

```python
import pycauset as pc

S = pc.symmetric([ [2.0, 1.0], [1.0, 5.0] ])   # -> pycauset.SymmetricMatrix
```

The factory checks that the input satisfies $A = A^T$ and then stores only the upper
triangle. For anti-symmetric matrices use [[docs/functions/pycauset.antisymmetric.md|pycauset.antisymmetric]]
(which returns a [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]]).

The raw native constructor `pycauset.SymmetricMatrix(n)` allocates an empty $N \times N$
symmetric matrix; fill the upper triangle with `set(i, j, value)`.

## Indexing

Element access uses NumPy-style indexing:

*   Read: `x = A[i, j]` (if `i > j`, reads the mirrored lower-triangle value)
*   Write: `A[i, j] = value` (if `i > j`, writes the mirrored lower-triangle value)

## `transpose()`

Returns the transpose. For a symmetric matrix the transpose is the matrix itself.

## Storage

Packed upper-triangular format. The storage size is approximately $N(N+1)/2$ elements,
about 2x smaller than dense for large $N$.

## See also

* [[docs/functions/pycauset.symmetric.md|pycauset.symmetric]]
* [[docs/classes/matrix/pycauset.AntiSymmetricMatrix.md|pycauset.AntiSymmetricMatrix]]
