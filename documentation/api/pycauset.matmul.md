# pycauset.matmul

```python
pycauset.matmul(a, b)
```

Perform matrix multiplication.

If both inputs are [[pycauset.TriangularBitMatrix]] instances, an optimized C++ implementation is used, returning an [[pycauset.IntegerMatrix]].
Otherwise, a generic multiplication is performed, returning a [[pycauset.Matrix]].

## Parameters

*   **a** (*[[pycauset.Matrix]]*): Left operand.
*   **b** (*[[pycauset.Matrix]]*): Right operand.

## Returns

*   **IntegerMatrix** or **Matrix**: The result of the multiplication.
