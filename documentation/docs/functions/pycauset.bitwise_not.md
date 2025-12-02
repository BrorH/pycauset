# pycauset.bitwise_not

```python
pycauset.bitwise_not(matrix: Any) -> Any
```

Compute the bitwise inversion (NOT) of a matrix.

## Parameters

*   **matrix** (*MatrixBase* or *array-like*): The matrix to invert. Must be a `TriangularBitMatrix`, `IntegerMatrix`, or similar bit-supporting type.

## Returns

*   **MatrixBase** or *array-like*: A new matrix with inverted bits.

## Raises

*   **TypeError**: If the object does not support bitwise inversion.
