# pycauset.bitwise_xnor

```python
pycauset.bitwise_xnor(a, b)    # ~(a ^ b)
```

Element-wise XNOR, `~(a ^ b)` (no NumPy equivalent). For matrices and vectors of dtype `bool` or an integer dtype.

Works for every structure and view (results are dense); operands of equal shape.

dtype rule (NumPy): `bool` op `bool` is `bool`; integer operands promote as NumPy does
(`int8` op `uint8` is `int16`; `bool` op `int32` is `int32`). Float and complex operands
raise `TypeError`, as in NumPy.

## Parameters

*   **a**, **b** (matrix or vector): operands of equal shape.

## Returns

A dense matrix (or vector) of the result dtype.

## Example

```python
import pycauset as pc
import numpy as np

a = pc.matrix(np.array([[1, 2], [3, 4]], dtype=np.int8))
b = pc.matrix(np.array([[3, 3], [3, 3]], dtype=np.int8))
print(pc.bitwise_xnor(a, b))
```

## See also

[[docs/functions/pycauset.bitwise_and.md|bitwise_and]] · [[docs/functions/pycauset.bitwise_or.md|bitwise_or]] · [[docs/functions/pycauset.bitwise_xor.md|bitwise_xor]] · [[docs/functions/pycauset.bitwise_nand.md|bitwise_nand]] · [[docs/functions/pycauset.bitwise_nor.md|bitwise_nor]] · [[docs/functions/pycauset.bitwise_not.md|bitwise_not]]
