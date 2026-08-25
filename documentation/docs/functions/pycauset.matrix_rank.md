# pycauset.matrix_rank

```python
pycauset.matrix_rank(a, tol=None)
```

Returns the numerical rank of a matrix (the number of singular values above `tol`).

## Parameters

*   **a** (matrix): The input matrix.
*   **tol** (float, optional): Singular-value threshold. Defaults to NumPy's default.

## Returns

An integer rank.

## Notes

Structural shortcuts avoid an SVD:

*   `is_zero` returns `0`.
*   `is_identity` returns `min(rows, cols)`.
*   `is_diagonal` / triangular returns the count of non-zero diagonal entries.

Otherwise a NumPy SVD-based rank is computed.

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 0.0], [0.0, 0.0] ])
pc.matrix_rank(A)   # 1
```

## See also

* [[docs/functions/pycauset.svdvals.md|pycauset.svdvals]]
