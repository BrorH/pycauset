# pycauset.I

```python
class pycauset.I(x)
class pycauset.I(n: int)
class pycauset.I(rows: int, cols: int)
```

Alias for [[pycauset.IdentityMatrix]].

`pycauset.I(x)` accepts the same input forms as [[docs/functions/pycauset.identity.md|pycauset.identity]]:

- `N` (int) -> $N \times N$
- `[rows, cols]` -> `rows × cols`
- matrix -> `(x.rows(), x.cols())`
- vector -> $N \times N$ where $N = x.size()$

## Usage

```python
import pycauset

# Create a 1000x1000 identity matrix
identity = pycauset.I(1000)

# Create a rectangular identity-like matrix (ones on the diagonal up to min(rows, cols))
rect_id = pycauset.I(3, 5)

# Also supported:
rect_id2 = pycauset.I([3, 5])

A = pycauset.FloatMatrix(2, 4)
IA = pycauset.I(A)          # 2x4

v = pycauset.IntegerVector(7)
Iv = pycauset.I(v)          # 7x7
```
