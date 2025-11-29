```
class pycauset.TriangularFloatMatrix
```
A memory-mapped matrix containing double-precision floating point values. This class is returned by [[pycauset.compute_k]].

Inherits from [[pycauset.TriangularMatrix]].

### Methods:
- `get(i, j)`: Returns the value at row `i` and column `j`.
- `size()`: Returns the dimension $N$ of the matrix.
- `close()`: Closes the memory-mapped file handle.

### Properties:
- `shape`: Tuple `(N, N)`.
