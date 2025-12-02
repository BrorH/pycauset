# pycauset.matmul

```python
pycauset.matmul(a: MatrixBase, b: MatrixBase, saveas: str = "") -> MatrixBase
```

Perform matrix multiplication.

If both inputs are [[pycauset.TriangularBitMatrix]] instances, an optimized C++ implementation is used, returning a [[pycauset.TriangularIntegerMatrix]].
Otherwise, a generic multiplication is performed.

## Parameters

*   **a** (*MatrixBase*): Left operand.
*   **b** (*MatrixBase*): Right operand.
*   **saveas** (*str*, optional): Path to save the result matrix. If not provided, a temporary file is used.

## Returns

*   **MatrixBase**: The result of the multiplication. The specific type depends on the input types.
