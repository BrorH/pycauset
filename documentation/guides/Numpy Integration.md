# NumPy Integration Guide

`pycauset` is designed to work seamlessly with the Python scientific stack, particularly NumPy. While `pycauset` uses its own optimized storage (RAM or disk-backed) for handling massive datasets, it provides smooth interoperability with NumPy arrays for convenience and flexibility.

## Converting NumPy Arrays to PyCauset

You can convert NumPy arrays into `pycauset` objects using the [[pycauset.Matrix]] and [[pycauset.Vector]] factory functions. These functions automatically detect the data type of the NumPy array and create the corresponding optimized object.

```python
import numpy as np
import pycauset as pc

# Convert 1D NumPy array to Vector
arr_1d = np.array([1.0, 2.0, 3.0])
vec = pc.Vector(arr_1d)  # Returns [[pycauset.FloatVector]]

# Convert 2D NumPy array to Matrix
arr_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
mat = pc.Matrix(arr_2d)  # Returns [[pycauset.IntegerMatrix]]

# Convert Boolean array
arr_bool = np.array([True, False], dtype=bool)
vec_bool = pc.Vector(arr_bool)  # Returns [[pycauset.BitVector]]
```

**Note**: This operation creates a **copy** of the data. Depending on the size and the configured memory threshold, the new object will be stored in RAM or on disk, see [[User Guide#Storage Management]]

## Converting PyCauset Objects to NumPy

All `pycauset` Matrix and Vector classes implement the NumPy array protocol (`__array__`). This means you can pass any `pycauset` object directly to `np.array()` or any function that expects an array-like object.

```python
v = pc.Vector([1, 2, 3])

# Convert to NumPy array
arr = np.array(v)

# Use in NumPy functions
mean_val = np.mean(v)
std_val = np.std(v)
```

**Note**: This operation loads the entire dataset into memory as a standard NumPy array. Be careful when doing this with very large disk-backed matrices that exceed your RAM capacity.

## Mixed Arithmetic

You can perform arithmetic operations directly between `pycauset` objects and NumPy arrays. `pycauset` handles the interoperability automatically.

### Vector + NumPy Array

```python
v = pc.Vector([1, 2, 3])
arr = np.array([10, 20, 30])

# Result is a pycauset Vector (operation happens in C++ backend)
result = v + arr  # [11, 22, 33]
```

### Matrix @ NumPy Vector

You can use NumPy arrays as operands in matrix multiplication.

```python
M = pc.Matrix([[1, 0], [0, 1]]) # Identity
v_np = np.array([5.0, 6.0])

# Result is a pycauset Vector
v_result = M @ v_np  # [5.0, 6.0]
```

## Performance Considerations

*   **PyCauset as Primary**: When you perform operations like `pycauset_obj + numpy_obj`, `pycauset` attempts to handle the operation. The NumPy array is temporarily converted to a `pycauset` object (backed by RAM or a temporary file), and the operation runs using the optimized C++ backend. The result is a new `pycauset` object.
*   **NumPy as Primary**: If you use a NumPy function like `np.add(pycauset_obj, numpy_obj)`, NumPy will convert the `pycauset` object to an in-memory array first. This might be slower and memory-intensive for large datasets.

**Best Practice**: For massive datasets, stick to `pycauset` native operations and objects as much as possible, only converting to NumPy for small results or specific analysis steps that `pycauset` doesn't yet support.
