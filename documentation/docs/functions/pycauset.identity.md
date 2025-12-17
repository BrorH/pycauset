# pycauset.identity

```python
pycauset.identity(x)
```

Creates an identity-like matrix (ones on the diagonal, zeros elsewhere) with shape derived from `x`.

This is a convenience factory around [[pycauset.IdentityMatrix]] and supports multiple input forms.

## Parameters

`x` can be:

1. **Integer `N`**
   - Returns an $N \times N$ identity matrix.

2. **Shape sequence `[rows, cols]`**
   - Returns a `rows × cols` identity-like matrix.
   - The diagonal is filled with ones up to $\min(rows, cols)$.

3. **A Matrix or Vector**
   - If `x` is a matrix, returns an identity-like matrix with shape `(x.rows(), x.cols())`.
   - If `x` is a vector, returns an $N \times N$ identity matrix where $N = x.size()$.

## Examples

```python
import pycauset as pc

# 1) Integer input
I5 = pc.identity(5)            # 5x5

# 2) Rectangular shape input
I35 = pc.identity([3, 5])      # 3x5

# 3) Matrix input
A = pc.FloatMatrix(2, 4)
IA = pc.identity(A)            # 2x4

# 3) Vector input
v = pc.IntegerVector(7)
Iv = pc.identity(v)            # 7x7
```

## See also

* [[docs/classes/matrix/pycauset.IdentityMatrix.md|pycauset.IdentityMatrix]]
* [[docs/functions/pycauset.I.md|pycauset.I]]
