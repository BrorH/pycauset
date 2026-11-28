# pycauset.norm

```python
pycauset.norm(x, ord=None) -> float
```

Norm of a vector or matrix with NumPy's `ord` values, computed in the engine for every dtype,
structure and view (double precision).

## Parameters

- `x`: a vector or matrix (array-likes become a dense matrix).
- `ord`:
  - matrices: `None` / `'fro'` (Frobenius), `'nuc'` (sum of singular values), `2` / `-2`
    (largest / smallest singular value), `1` / `-1` (max / min column sum of `|x|`),
    `inf` / `-inf` (max / min row sum);
  - vectors: `None` (2-norm), any real `p` (`(sum |x|^p)^(1/p)`), `inf` / `-inf` (max / min
    `|x|`), `0` (number of non-zeros).

## Returns

- `float`.

## Raises

- `ValueError`: an `ord` NumPy does not define for that input. For an empty matrix, as in
  NumPy: the orders that take a maximum or a sum (`fro`, `nuc`, `1`, `2`, `inf`) give 0; the
  minimum of nothing raises (`-2`; `-1` with no columns, `-inf` with no rows).

## Examples

```python
import pycauset as pc

v = pc.vector([3.0, 4.0])
assert pc.norm(v) == 5.0
assert pc.norm(v, 1) == 7.0

A = pc.matrix([[3.0, 4.0], [0.0, 0.0]])
assert pc.norm(A) == 5.0
assert pc.norm(A, 1) == 4.0
```
