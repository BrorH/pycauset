# pycauset.matmul

```python
pycauset.matmul(a, b)
```

Perform matrix multiplication.

For native objects, this dispatches to optimized C++ implementations.

This function is NumPy-like:

- matrix-matrix: `(m, k) @ (k, n) -> (m, n)`
- matrix-vector: `(m, k) @ (k,) -> (m,)`
- vector-matrix: `(k,) @ (k, n) -> (1, n)` (row-vector semantics)
- vector-vector: `(k,) @ (k,) -> scalar` (dot)

## Parameters

*   **a** (*MatrixBase or VectorBase*): Left operand.
*   **b** (*MatrixBase or VectorBase*): Right operand.

Shape rule: `a.cols() == b.rows()`.

## Returns

*   **MatrixBase or VectorBase or scalar**: The result. The specific type and shape depend on input ranks.

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
*   [[guides/Matrix Guide|Matrix Guide]]
