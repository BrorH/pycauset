# pycauset.asarray

```python
pycauset.asarray(a, dtype=None)
```

A PyCauset vector (1-D input) or matrix (2-D input), as NumPy's `asarray`: a PyCauset object is
returned as it is (no copy) unless another `dtype` is asked for; NumPy arrays, lists and tuples
are imported, with the same rules as [[docs/functions/pycauset.matrix.md|pycauset.matrix]] and
[[docs/functions/pycauset.vector.md|pycauset.vector]] (one import path; byte order and strides of
any NumPy array are normalised).

## Parameters

*   **a** (*array-like*): a PyCauset object, a NumPy array, or nested lists / tuples; 1-D or 2-D.
*   **dtype** (*optional*): the dtype of the result (default: the input's).

## Returns

*   **VectorBase or MatrixBase**: The native PyCauset object matching the input shape and dtype.

The exact class depends on the input:

| Input | Result |
| :--- | :--- |
| 1D `float64` | `DenseVector` (dtype `float64`) |
| 1D `int64` | `DenseVector` (dtype `int64`) |
| 1D `bool` | `DenseVector` (dtype `bool`) |
| 2D `float64` | `DenseMatrix` (dtype `float64`) |
| 2D `int64` | `DenseMatrix` (dtype `int64`) |
| 2D `bool` | `DenseMatrix` (dtype `bool`) |
| 2D `complex128` | `DenseMatrix` (dtype `complex_float64`) |

Other NumPy dtypes map to the corresponding native class (e.g. `float32` →
`DenseMatrix` (dtype `float32`), `uint8` → `DenseMatrix` (dtype `uint8`)).

## Exceptions

*   **ValueError**: the input is not 1-D or 2-D.
*   **TypeError**: a dtype PyCauset does not store.

## Examples

```python
import numpy as np
import pycauset as pc

arr = np.arange(6).reshape(2, 3)   # int64 2D
M = pc.asarray(arr)
assert type(M).__name__ == "DenseMatrix" and M.dtype == "int64"
assert M.shape == (2, 3)

b = np.full((2, 2), True, dtype=bool)   # a 2x2 boolean array
B = pc.asarray(b)                       # bit-packed boolean matrix
assert type(B).__name__ == "DenseMatrix" and B.dtype == "bool"
```

## See also

*   [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
*   [[docs/functions/pycauset.vector.md|pycauset.vector]]
*   [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]]
*   [[guides/Numpy Integration.md|NumPy Integration]]
