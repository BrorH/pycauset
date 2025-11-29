```
class pycauset.IntegerMatrix
```
A read-only memory-mapped matrix containing integer values. This class is typically returned by matrix multiplication operations like [[pycauset.matmul]].

Inherits from [[pycauset.Matrix]].

### Methods:
- `get(i, j)`: Returns the integer value at row `i` and column `j`.
- `size()`: Returns the dimension $N$ of the matrix.

### Properties:
- `shape`: Tuple `(N, N)`.
