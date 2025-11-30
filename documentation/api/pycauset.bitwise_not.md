# pycauset.bitwise_not

```python
pycauset.bitwise_not(matrix)
```

Computes the bitwise inversion (NOT) of a matrix. This is equivalent to using the `~` operator (e.g. `~matrix`).

*   For **TriangularBitMatrix**: Flips every bit in the upper triangle ($0 \to 1, 1 \to 0$).
*   For **IntegerMatrix**: Performs bitwise NOT on each 32-bit integer element.
*   For **FloatMatrix** and **TriangularFloatMatrix**: Performs bitwise NOT on the raw 64-bit floating-point representation. Note that this operation is generally not mathematically meaningful for floats but is provided for structural completeness.

## Parameters

*   **matrix** (*MatrixBase*): The matrix to invert.

## Returns

*   **MatrixBase**: A new matrix of the same type as the input, with inverted bits.
