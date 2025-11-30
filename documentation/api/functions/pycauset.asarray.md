# pycauset.asarray

```python
pycauset.asarray(obj)
```

Convert the input to a `pycauset` array (Matrix or Vector).

This function is similar to `numpy.asarray`. It detects the input data type and structure (1D or 2D) and creates the corresponding disk-backed `pycauset` object.

*   **1D Input**: Creates a `Vector` (`FloatVector`, `IntegerVector`, or `BitVector`).
*   **2D Input**: Creates a `Matrix` (`FloatMatrix`, `IntegerMatrix`, or `DenseBitMatrix`).

**Note**: This function always creates a new persistent object on disk, effectively copying the data from memory to disk.

## Parameters

*   **obj** (*array_like*): Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.

## Returns

*   **Vector** or **Matrix**: The `pycauset` representation of the input data.

## Examples

```python
import pycauset
import numpy as np

# Convert NumPy array to Vector
arr = np.array([1, 2, 3])
v = pycauset.asarray(arr)  # IntegerVector

# Convert list of lists to Matrix
data = [[1.0, 0.0], [0.0, 1.0]]
m = pycauset.asarray(data) # FloatMatrix
```
