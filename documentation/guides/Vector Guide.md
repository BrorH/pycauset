# Vector Guide

`pycauset` introduces efficient vectors that integrate seamlessly with the matrix operations. Vectors are stored in RAM for small sizes (behaving like NumPy arrays) and automatically spill to disk for massive datasets.

## Creating Vectors

You can create vectors using the [[pycauset.Vector]] factory function. It automatically selects the most efficient storage backend based on your data.

```python
from pycauset import Vector

# Create a float vector of size 1000 (initialized to 0.0)
v1 = Vector(1000)

# Create an integer vector from a list
v2 = Vector([1, 2, 3, 4, 5])

# Create a bit-packed boolean vector (1 bit per element)
v3 = Vector([True, False, True, True])
```

## Vector Arithmetic

Vectors support standard arithmetic operations. These operations are performed element-wise and are optimized for large datasets.

### Addition

```python
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
v3 = v1 + v2  # [5, 7, 9]
```

### Subtraction

```python
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
v3 = v2 - v1  # [3, 3, 3]
```

### Scalar Addition

You can add a scalar to every element of a vector.

```python
v = Vector([1, 2, 3])
v_plus_5 = v + 5  # [6, 7, 8]
v_plus_5_reverse = 5 + v  # [6, 7, 8]
```

### Scalar Multiplication

```python
v = Vector([1, 2, 3])
v_scaled = v * 2.0  # [2.0, 4.0, 6.0]
```

### Dot Product

You can compute the dot product of two vectors using [[pycauset.dot]] or the `dot` method.

```python
from pycauset import dot

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

result = dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32.0
# OR
result = v1.dot(v2)
```

## Transposition & Matrix Operations

`pycauset` supports vector transposition and various forms of multiplication using the `@` operator (matrix multiplication), mirroring `numpy` behavior.

### Transposition (`.T`)

You can transpose a vector using the `.T` property.
*   **Column Vector** (Default): Shape `(N,)`.
*   **Row Vector** (`v.T`): Shape `(1, N)`.

```python
v = Vector([1, 2, 3])
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
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

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
v1 = Vector([1, 2, 3]) # Column
v2 = Vector([4, 5, 6]) # Column

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
M = pycauset.FloatMatrix(3)
v = Vector([1, 1, 1])

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
v = pycauset.asarray(arr)
```

### Mixed Operations
You can add, subtract, or multiply vectors with NumPy arrays directly. The result remains a persistent `pycauset` vector.

```python
v = Vector([1, 2, 3])
arr = np.array([10, 10, 10])
v_new = v + arr # [11, 12, 13]
```

## Persistence

Like matrices, vectors are backed by storage (RAM or disk). You can save them permanently using [[pycauset.save]].

```python
from pycauset import save, load

v = Vector([1, 2, 3])
save(v, "my_vector.pycauset")

v_loaded = load("my_vector.pycauset")
```

## Mixed Types

Operations between different vector types (e.g., [[pycauset.IntegerVector]] + [[pycauset.FloatVector]]) are supported. The result is typically promoted to a [[pycauset.FloatVector]] (DenseVector<double>) to ensure precision.

```python
v_int = Vector([1, 2], dtype="int")
v_float = Vector([0.5, 0.5], dtype="float")

v_sum = v_int + v_float  # [1.5, 2.5] (FloatVector)
```
