# pycauset.matrix_rank

```python
pycauset.matrix_rank(a, tol=None)
```

Returns the numerical rank of a matrix (the number of singular values above `tol`).

## Parameters

*   **a** (matrix): The input matrix.
*   **tol** (float, optional): Singular-value threshold. Defaults to NumPy's default tolerance.

## Returns

An integer rank.

## Notes

The number of singular values of the engine's SVD above `tol`; the default is NumPy's,
`max(s) * max(m, n) * eps`. Every dtype, structure and view.

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 0.0], [0.0, 0.0] ])
pc.matrix_rank(A)   # 1
```

## See also

* [[docs/functions/pycauset.svdvals.md|pycauset.svdvals]]
