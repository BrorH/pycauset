# pycauset.divide

```python
pycauset.divide(a, b) -> Any
```

Elementwise division.

This is a convenience wrapper around the `/` operator and follows **NumPy-style 2D broadcasting** for matrix inputs.

## Parameters

- `a`: Left operand (typically a matrix).
- `b`: Right operand (matrix, scalar, or NumPy array depending on the operator overload).

## Returns

- A new `pycauset` object containing the elementwise division result.

## Broadcasting rules (2D)

Two shapes `(a_rows, a_cols)` and `(b_rows, b_cols)` are compatible if each dimension is either equal or one of them is `1`. The result shape is:

- `(max(a_rows, b_rows), max(a_cols, b_cols))`

When mixing a matrix with a **1D NumPy array** in an elementwise operation, the array is treated as a **row vector** of shape `(1, n)`.

## Dtype behavior

- If either operand is float/complex, the result is a float/complex type.
- If neither operand is float/complex (e.g. int/uint/bit), the result promotes to a float dtype based on the current promotion precision mode:
  - `lowest` (default): `float32`
  - `highest`: `float64`

## Exceptions

- Raises `TypeError` if the operand types are not supported by the `/` operator.
- Raises `ValueError` / `RuntimeError` on invalid shapes or unsupported dtype combinations.

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

- [[docs/functions/pycauset.precision_mode.md|pycauset.precision_mode]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
