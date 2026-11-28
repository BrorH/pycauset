# pycauset.divide

```python
pycauset.divide(a, b) -> Any
```

Elementwise division.

The function form of the `/` operator, with **NumPy-style 2D broadcasting**. Matrices,
vectors, NumPy arrays and nested lists are accepted: array-like operands are imported into the
engine first (an ndarray pair used to be divided by NumPy). One operand may be a scalar; two
scalars raise `TypeError`.

## Parameters

- `a`: Left operand (typically a matrix).
- `b`: Right operand (matrix, vector, array-like or scalar).

## Returns

- A new `pycauset` object containing the elementwise division result.

## Broadcasting rules (2D)

Two shapes `(a_rows, a_cols)` and `(b_rows, b_cols)` are compatible if each dimension is either equal or one of them is `1`. The result shape is:

- `(max(a_rows, b_rows), max(a_cols, b_cols))`

When mixing a matrix with a **1D NumPy array** in an elementwise operation, the array is treated as a **row vector** of shape `(1, n)`.

## Dtype behavior

- If either operand is float/complex, the result is a float/complex type.
- If neither operand is float/complex (e.g. int/uint/bit), the result is a float that can
  hold both operand dtypes: `float32` for bit/8-bit/16-bit integers, `float64` for 32/64-bit
  integers. This holds for a scalar divisor too (`int8 M / 1` is float32). NumPy gives
  float64 for every integer division.

## Exceptions

- Raises `TypeError` if the operand types are not supported by the `/` operator, or both are
  scalars.
- Raises `ValueError` on shapes that do not broadcast.

## Examples

```python
import numpy as np
import pycauset as pc

A = pc.zeros((3, 4), dtype="float64")
row = np.arange(4, dtype=np.float64)

B = pc.divide(A, row)   # broadcasts (3,4) / (1,4)
C = A / row             # equivalent
```

## See also

- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
