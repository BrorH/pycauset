# Vector Guide

`pycauset` introduces efficient vectors that integrate seamlessly with the matrix operations. Vectors are stored in RAM for small sizes (behaving like NumPy arrays) and automatically spill to disk for massive datasets.

## Creating Vectors

You can create vectors using the [[pycauset.Vector]] factory function. It automatically selects the most efficient storage backend based on your data.

You can also explicitly control storage using `dtype`.

Supported dtype strings (recommended):

- Bit/boolean: `"bit"`, `"bool"`, `"bool_"`
- Signed integers: `"int8"`, `"int16"`, `"int32"`, `"int64"`
- Unsigned integers: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
- Floats: `"float16"`, `"float32"`, `"float64"`
- Complex floats: `"complex_float16"`, `"complex_float32"`, `"complex_float64"`

Notes:

- `"int"` normalizes to `"int32"`; `"float"` normalizes to `"float64"`; `"uint"` normalizes to `"uint32"`.
- Complex is limited to complex floats.
- Exact op coverage is declared in the support matrix (see `documentation/internals/DType System.md`).

```python
import pycauset as pc

# Create a float vector of size 1000 (initialized to 0.0)
v1 = pc.Vector(1000)

# Create an integer vector from a list
v2 = pc.Vector([1, 2, 3, 4, 5])

# Explicit widths / unsigned
v_i8 = pc.Vector([1, 2, 3], dtype="int8")
v_u64 = pc.Vector([1, 2, 3], dtype="uint64")

# Complex float vectors
v_c = pc.Vector([1+2j, 3-4j], dtype="complex_float32")

# Create a bit-packed boolean vector (1 bit per element)
v3 = pc.Vector([True, False, True, True])
```

## Unit Vectors

A `UnitVector` is a specialized vector type representing a standard basis vector (a vector with a single `1` at a specific index and `0` everywhere else). It is highly optimized for storage and arithmetic operations.

```python
import pycauset as pc

# Create a unit vector of size 1000 with the 1 at index 5
u = pc.UnitVector(1000, 5)

print(u[5]) # 1.0
print(u[0]) # 0.0
```

**Benefits:**
*   **O(1) Storage**: Only stores the index and size.
*   **O(1) Arithmetic**: Operations like dot products or addition with other unit vectors are computed instantly without iterating over the full size.

## Vector Arithmetic

Vectors support standard arithmetic operations. These operations are performed element-wise and are optimized for large datasets.

### Optimized Operations

Arithmetic involving `UnitVector` is special-cased for performance.
*   `UnitVector` + `UnitVector`: Returns a sparse result if possible.
*   `Vector` + `UnitVector`: Only updates the single active element.
*   `Vector` . `UnitVector`: Returns the element at the active index (O(1)).

### Addition

```python
v1 = pc.Vector([1, 2, 3])
v2 = pc.Vector([4, 5, 6])
v3 = v1 + v2  # [5, 7, 9]
```

### Subtraction

```python
v1 = pc.Vector([1, 2, 3])
v2 = pc.Vector([4, 5, 6])
v3 = v2 - v1  # [3, 3, 3]
```

### Scalar Addition

You can add a scalar to every element of a vector.

```python
v = pc.Vector([1, 2, 3])
v_plus_5 = v + 5  # [6, 7, 8]
v_plus_5_reverse = 5 + v  # [6, 7, 8]
```

### Scalar Multiplication

```python
v = pc.Vector([1, 2, 3])
v_scaled = v * 2.0  # [2.0, 4.0, 6.0]
```

### Dot Product

You can compute the dot product of two vectors using [[pycauset.dot]] or the `dot` method.

```python
import pycauset as pc

v1 = pc.Vector([1, 2, 3])
v2 = pc.Vector([4, 5, 6])

result = pc.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32.0
# OR
result = v1.dot(v2)
```

### Cross Product

The cross product is defined for 3D vectors.

```python
v1 = pc.Vector([1, 0, 0])
v2 = pc.Vector([0, 1, 0])
v3 = v1.cross(v2)  # [0, 0, 1]
```

You can compute the cross product of two 3D vectors using [[pycauset.cross]].

```python
import pycauset as pc

v1 = pc.Vector([1, 0, 0])
v2 = pc.Vector([0, 1, 0])

# v1 x v2 = [0, 0, 1]
v3 = pc.cross(v1, v2)
```

**Note**: The cross product is only defined for vectors of size 3. Passing vectors of other sizes will raise a `ValueError`.

## Transposition & Matrix Operations

`pycauset` supports vector transposition and various forms of multiplication using the `@` operator (matrix multiplication), mirroring `numpy` behavior.

### Transposition (`.T`)

You can transpose a vector using the `.T` property.
*   **Column Vector** (Default): Shape `(N,)`.
*   **Row Vector** (`v.T`): Shape `(1, N)`.

```python
v = pc.Vector([1, 2, 3])
print(v.shape)    # (3,)

vt = v.T
print(vt.shape)   # (1, 3)

v_orig = vt.T     # Back to column vector
print(v_orig.shape) # (3,)
```

**Note**: Transposing creates a new persistent object (a new file) with the `is_transposed` flag flipped. It does *not* modify the original vector.

### Inner Product (Dot Product)

The inner product produces a scalar.

```python
v1 = pc.Vector([1, 2, 3])
v2 = pc.Vector([4, 5, 6])

# Row @ Column -> Scalar
scalar = v1.T @ v2  # 32.0

# Column @ Column (Numpy behavior for 1D arrays)
scalar = v1 @ v2    # 32.0
```

**Type Safety**:
*   [[pycauset.IntegerVector]] @ [[pycauset.IntegerVector]] -> `int`
*   [[pycauset.BitVector]] @ [[pycauset.BitVector]] -> `int`
*   Any [[pycauset.FloatVector]] operand -> `float`

### Outer Product

The outer product produces a matrix.

```python
v1 = pc.Vector([1, 2, 3]) # Column
v2 = pc.Vector([4, 5, 6]) # Column

# Column @ Row -> Matrix (N x N)
M = v1 @ v2.T 
# M is:
# [[ 4,  5,  6],
#  [ 8, 10, 12],
#  [12, 15, 18]]
```

**Type Safety**:
*   [[pycauset.BitVector]] @ [[pycauset.BitVector]].T -> [[pycauset.DenseBitMatrix]] (Logical AND)
*   [[pycauset.IntegerVector]] @ [[pycauset.IntegerVector]].T -> [[pycauset.IntegerMatrix]]
*   Others -> [[pycauset.FloatMatrix]]

### Matrix-Vector Multiplication

You can multiply matrices and vectors.

```python
M = pc.Matrix(3, dtype=pc.float64)
v = pc.Vector([1, 1, 1])

# Matrix @ Column Vector -> Column Vector
v_new = M @ v 

# Row Vector @ Matrix -> Row Vector
v_row = v.T @ M
```

## NumPy Compatibility

Vectors are fully compatible with NumPy.

### To NumPy
You can convert any vector to a NumPy array:
```python
import numpy as np
arr = np.array(v)
```

### From NumPy
You can create vectors from NumPy arrays using [[pycauset.asarray]]:
```python
arr = np.array([1.0, 2.0, 3.0])
v = pc.asarray(arr)
```

### Mixed Operations
You can add, subtract, or multiply vectors with NumPy arrays directly. The result remains a persistent `pycauset` vector.

```python
v = pc.Vector([1, 2, 3])
arr = np.array([10, 10, 10])
v_new = v + arr # [11, 12, 13]
```

## Persistence

Like matrices, vectors are backed by storage (RAM or disk). You can save them permanently using [[pycauset.save]].

```python
import pycauset as pc

v = pc.Vector([1, 2, 3])
pc.save(v, "my_vector.pycauset")

v_loaded = pc.load("my_vector.pycauset")
```

## Mixed Types

Operations between different vector dtypes (e.g., integer + float) are supported. The result kind follows the fundamental-kind rule from `documentation/internals/DType System.md` (if a float participates, the result kind is float).

```python
v_int = pc.Vector([1, 2], dtype=pc.int32)
v_float = pc.Vector([0.5, 0.5], dtype=pc.float64)

v_sum = v_int + v_float  # [1.5, 2.5] (FloatVector)
```
