# pycauset.svdvals

```python
pycauset.svdvals(a)
```

Returns the singular values of a matrix (the `S` vector of an SVD), in descending order.

## Parameters

*   **a** (matrix): The input matrix.

## Returns

A vector of singular values, descending.

## Notes

Implemented via NumPy SVD with `compute_uv=False`.

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 0.0], [0.0, 2.0] ])
pc.svdvals(A)   # [2.0, 1.0]
```

## See also

* [[docs/functions/pycauset.svd.md|pycauset.svd]]
* [[docs/functions/pycauset.matrix_rank.md|pycauset.matrix_rank]]
