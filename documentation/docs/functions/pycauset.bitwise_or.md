# pycauset.bitwise_or

```python
pycauset.bitwise_or(a, b)    # a | b
```

Element-wise bitwise OR (`a | b`). For matrices and vectors of dtype `bool` or an integer dtype.

Works for every structure and view (results are dense). Shapes broadcast as in NumPy (a
vector is a row). A scalar operand follows NumPy 2: a Python `bool` or `int` takes the integer
operand's dtype and must fit it (`OverflowError` otherwise; with a `bool` operand a Python int
gives `int64`); a NumPy scalar or 0-d array brings its own dtype.

dtype rule (NumPy): `bool` op `bool` is `bool`; integer operands promote as NumPy does
(`int8` op `uint8` is `int16`; `bool` op `int32` is `int32`). Float and complex operands
raise `TypeError`, as in NumPy.

## Parameters

*   **a**, **b** (matrix, vector, NumPy array or scalar; at least one not a scalar): the
    operands, broadcast together.

## Returns

A dense matrix (or vector) of the result dtype.

## Example

```python
import pycauset as pc
import numpy as np

a = pc.matrix(np.array([[1, 2], [3, 4]], dtype=np.int8))
b = pc.matrix(np.array([[3, 3], [3, 3]], dtype=np.int8))
print(pc.bitwise_or(a, b))
```

## See also

[[docs/functions/pycauset.bitwise_and.md|bitwise_and]] · [[docs/functions/pycauset.bitwise_xor.md|bitwise_xor]] · [[docs/functions/pycauset.bitwise_nand.md|bitwise_nand]] · [[docs/functions/pycauset.bitwise_nor.md|bitwise_nor]] · [[docs/functions/pycauset.bitwise_xnor.md|bitwise_xnor]] · [[docs/functions/pycauset.bitwise_not.md|bitwise_not]]
