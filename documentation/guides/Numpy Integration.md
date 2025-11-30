# NumPy Integration Guide

`pycauset` is designed to work seamlessly with the Python scientific stack, particularly NumPy. While `pycauset` uses its own disk-backed storage for handling massive datasets, it provides smooth interoperability with NumPy arrays for convenience and flexibility.

## Converting NumPy Arrays to PyCauset

You can convert NumPy arrays into `pycauset` objects using the `pycauset.asarray()` function. This function automatically detects the data type and shape of the NumPy array and creates the corresponding `pycauset` Matrix or Vector.

```python
import numpy as np
import pycauset

# Convert 1D NumPy array to Vector
arr_1d = np.array([1.0, 2.0, 3.0])
vec = pycauset.asarray(arr_1d)  # Returns FloatVector

# Convert 2D NumPy array to Matrix
arr_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
mat = pycauset.asarray(arr_2d)  # Returns IntegerMatrix

# Convert Boolean array
arr_bool = np.array([True, False], dtype=bool)
vec_bool = pycauset.asarray(arr_bool)  # Returns BitVector
```

**Note**: This operation creates a **copy** of the data on disk. `pycauset` objects are persistent, whereas NumPy arrays are in-memory.

## Converting PyCauset Objects to NumPy

All `pycauset` Matrix and Vector classes implement the NumPy array protocol (`__array__`). This means you can pass any `pycauset` object directly to `np.array()` or any function that expects an array-like object.

```python
v = pycauset.Vector([1, 2, 3])

# Convert to NumPy array
arr = np.array(v)

# Use in NumPy functions
mean_val = np.mean(v)
std_val = np.std(v)
```

**Note**: This operation loads the entire dataset from disk into memory. Be careful when doing this with very large matrices that exceed your RAM capacity.

## Mixed Arithmetic

You can perform arithmetic operations directly between `pycauset` objects and NumPy arrays. `pycauset` handles the interoperability automatically.

### Vector + NumPy Array

```python
v = pycauset.Vector([1, 2, 3])
arr = np.array([10, 20, 30])

# Result is a pycauset Vector (operation happens in C++ backend)
result = v + arr  # [11, 22, 33]
```

### Matrix @ NumPy Vector

You can use NumPy arrays as operands in matrix multiplication.

```python
M = pycauset.Matrix([[1, 0], [0, 1]]) # Identity
v_np = np.array([5.0, 6.0])

# Result is a pycauset Vector
v_result = M @ v_np  # [5.0, 6.0]
```

## Performance Considerations

*   **PyCauset as Primary**: When you perform operations like `pycauset_obj + numpy_obj`, `pycauset` attempts to handle the operation. The NumPy array is temporarily converted to a `pycauset` object (backed by a temporary file), and the operation runs using the optimized C++ backend. The result is a new `pycauset` object.
*   **NumPy as Primary**: If you use a NumPy function like `np.add(pycauset_obj, numpy_obj)`, NumPy will convert the `pycauset` object to an in-memory array first. This might be slower and memory-intensive for large datasets.

**Best Practice**: For massive datasets, stick to `pycauset` native operations and objects as much as possible, only converting to NumPy for small results or specific analysis steps that `pycauset` doesn't yet support.
