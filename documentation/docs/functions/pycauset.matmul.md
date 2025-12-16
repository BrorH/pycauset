# pycauset.matmul

```python
pycauset.matmul(a: MatrixBase, b: MatrixBase) -> MatrixBase
```

Perform matrix multiplication.

If both inputs are [[pycauset.TriangularBitMatrix]] instances, an optimized C++ implementation is used, returning a [[pycauset.TriangularIntegerMatrix]].
Otherwise, a generic multiplication is performed.

## Parameters

*   **a** (*MatrixBase*): Left operand.
*   **b** (*MatrixBase*): Right operand.

## Returns

*   **MatrixBase**: The result of the multiplication. The specific type depends on the input types.
