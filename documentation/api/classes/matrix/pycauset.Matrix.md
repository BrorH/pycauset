# pycauset.Matrix

```python
pycauset.Matrix(source, dtype=None)
```

The `Matrix` class serves as a smart factory for creating matrix objects in `pycauset`. It automatically selects the most optimized backend implementation based on the input data or the specified `dtype`.

*   If `dtype` is specified, it forces the creation of a specific matrix type:
    *   `dtype=int` (or `np.int32`, `np.int64`): Creates an `IntegerMatrix` (or `TriangularIntegerMatrix` if applicable).
    *   `dtype=float` (or `np.float64`): Creates a `FloatMatrix` (or `TriangularFloatMatrix` if applicable).
    *   `dtype=bool` (or `np.bool_`): Creates a `DenseBitMatrix` (or `TriangularBitMatrix` if applicable).
*   If `dtype` is NOT specified, it infers the type from the data:
    *   Integers + Strictly Upper Triangular -> `TriangularIntegerMatrix`
    *   Integers + Dense -> `IntegerMatrix`
    *   Floats + Triangular -> `TriangularFloatMatrix`
    *   Floats + Dense -> `FloatMatrix`
    *   Fallback -> Generic Python `Matrix`

All returned matrix objects share a common interface and are backed by disk storage (except the generic Python fallback).

## Parameters

*   **source** (*int, list, or numpy.ndarray*): 
    *   If an `int` is provided, it creates a new empty (zero-filled) matrix of size $N \times N$.
    *   If a nested list/tuple or NumPy array is provided, it creates a matrix populated with that data.
*   **dtype** (*type, optional*): The desired data type (`int`, `float`, `bool`). If provided, it enforces the use of a specific C++ backend.

## Returns

*   **Matrix**: An instance of a matrix class (e.g., `FloatMatrix`, `IntegerMatrix`, `DenseBitMatrix`, or the generic Python `Matrix` fallback).

## Optimized Backends

The `Matrix` factory returns instances of specialized classes optimized for specific data types and structures. While these classes are not directly exposed for instantiation, they all share the common `Matrix` interface.

*   **`IntegerMatrix`**: Dense matrix storing 32-bit signed integers.
*   **`FloatMatrix`**: Dense matrix storing 64-bit floating-point numbers.
*   **`DenseBitMatrix`**: Dense matrix storing boolean values (1 bit per element).
*   **`TriangularIntegerMatrix`**: Strictly upper triangular matrix storing 32-bit integers.
*   **`TriangularFloatMatrix`**: Strictly upper triangular matrix storing 64-bit floats.
*   **`TriangularBitMatrix`**: Strictly upper triangular matrix storing boolean values (1 bit per element).

## Examples

```python
# Create a 5x5 empty integer matrix
M = pycauset.Matrix(5, dtype=int)

# Create a dense boolean matrix (DenseBitMatrix)
M_bool = pycauset.Matrix(5, dtype=bool)

# Create an integer matrix from a list
data = [[1, 2], [3, 4]]
M_int = pycauset.Matrix(data) 
# Returns an optimized IntegerMatrix

# Create a triangular matrix
tri_data = [[0, 1], [0, 0]]
M_tri = pycauset.Matrix(tri_data)
# Returns an optimized TriangularIntegerMatrix
```
