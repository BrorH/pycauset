# pycauset.Vector

```python
pycauset.Vector(source, dtype=None)
```

The `Vector` class serves as a smart factory for creating vector objects in `pycauset`. It automatically selects the most optimized backend implementation based on the input data or the specified `dtype`.

*   If `dtype` is specified, it forces the creation of a specific vector type:
    *   `dtype=int` (or `np.int32`, `np.int64`): Creates an `IntegerVector`.
    *   `dtype=float` (or `np.float64`): Creates a `FloatVector`.
    *   `dtype=bool` (or `np.bool_`): Creates a `BitVector`.
*   If `dtype` is NOT specified, it infers the type from the data:
    *   Integers -> `IntegerVector`
    *   Floats -> `FloatVector`
    *   Booleans -> `BitVector`

All returned vector objects share a common interface and are backed by disk storage.

## Parameters

*   **source** (*int, list, or numpy.ndarray*): 
    *   If an `int` is provided, it creates a new empty (zero-filled) vector of size $N$.
    *   If a list/tuple or NumPy array is provided, it creates a vector populated with that data.
*   **dtype** (*type, optional*): The desired data type (`int`, `float`, `bool`). If provided, it enforces the use of a specific C++ backend.

## Returns

*   **Vector**: An instance of a vector class (e.g., `FloatVector`, `IntegerVector`, `BitVector`).

## Optimized Backends

The `Vector` factory returns instances of specialized classes optimized for specific data types. While these classes are not typically instantiated directly, they all share the common `Vector` interface.

*   **`IntegerVector`**: Dense vector storing 32-bit signed integers.
*   **`FloatVector`**: Dense vector storing 64-bit floating-point numbers.
*   **`BitVector`**: Dense vector storing boolean values (1 bit per element).

## Examples

```python
import pycauset

# Create a vector of size 1000 initialized to zeros (floats by default if no dtype)
v1 = pycauset.Vector(1000, dtype=float)

# Create an integer vector from a list
v2 = pycauset.Vector([1, 2, 3, 4, 5])

# Create a bit vector from a boolean list
v3 = pycauset.Vector([True, False, True])

# Create from a NumPy array
import numpy as np
arr = np.array([1.5, 2.5, 3.5])
v4 = pycauset.Vector(arr)  # Creates FloatVector
```
