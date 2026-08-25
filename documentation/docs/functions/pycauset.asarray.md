# pycauset.asarray

```python
pycauset.asarray(a)
```

Convert a NumPy array into a native PyCauset matrix or vector.

This is the low-level NumPy-to-PyCauset conversion entry point. It maps the input
array's shape and dtype to the matching native class, so the result participates in
the full PyCauset storage model (memory-mapped backing files, bit-packed booleans,
caching, etc.).

For most workflows, prefer the data constructors [[docs/functions/pycauset.matrix.md|pycauset.matrix]] and
[[docs/functions/pycauset.vector.md|pycauset.vector]], which accept both NumPy arrays and nested
sequences. Use `pycauset.asarray` when you already hold a NumPy array and want the
native object directly.

## Parameters

*   **a** (*numpy.ndarray*): The input array. Must be 1D or 2D.

## Returns

*   **VectorBase or MatrixBase**: The native PyCauset object matching the input shape and dtype.

The exact class depends on the input:

| Input | Result |
| :--- | :--- |
| 1D `float64` | `FloatVector` |
| 1D `int64` | `Int64Vector` |
| 1D `bool` | `BitVector` |
| 2D `float64` | `FloatMatrix` |
| 2D `int64` | `Int64Matrix` |
| 2D `bool` | `DenseBitMatrix` |
| 2D `complex128` | `ComplexFloat64Matrix` |

Other NumPy dtypes map to the corresponding native class (e.g. `float32` →
`Float32Matrix`, `uint8` → `UInt8Matrix`).

## Exceptions

*   **TypeError**: Raised when the input is not a `numpy.ndarray` (e.g. a plain list),
    or when the array has more than two dimensions.

## Examples

```python
import numpy as np
import pycauset as pc

arr = np.arange(6).reshape(2, 3)   # int64 2D
M = pc.asarray(arr)
assert type(M).__name__ == "Int64Matrix"
assert M.shape == (2, 3)

b = np.full((2, 2), True, dtype=bool)   # a 2x2 boolean array
B = pc.asarray(b)                       # bit-packed boolean matrix
assert type(B).__name__ == "DenseBitMatrix"
```

## See also

*   [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
*   [[docs/functions/pycauset.vector.md|pycauset.vector]]
*   [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]]
*   [[guides/Numpy Integration.md|NumPy Integration]]
