# Vector Guide

Vectors work with the matrix operations. Small ones live in RAM (like NumPy arrays); large ones spill to a temporary memory-mapped file (`.tmp`).

## Creating Vectors

You can create vectors from data using **[[docs/functions/pycauset.vector.md|pycauset.vector]]**. For allocation by size, use **[[docs/functions/pycauset.zeros.md|pycauset.zeros]]** or **[[docs/functions/pycauset.empty.md|pycauset.empty]]** with an explicit `dtype`.

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
- For large NumPy inputs, you can pass `max_in_ram_bytes` to `pc.vector(...)` to force the constructor down the internal `native.asarray` import path once the estimated payload exceeds the cap, avoiding accidental full-RAM materialization when you prefer a backing file.

```python
import pycauset as pc

# Allocate a float vector of size 1000 (initialized to 0.0)
v1 = pc.zeros(1000, dtype="float64")

# Create an integer vector from a list
v2 = pc.vector([1, 2, 3, 4, 5])

# Explicit widths / unsigned
v_u64 = pc.vector([1, 2, 3], dtype="uint64")

# Complex float vectors
v_c = pc.vector([1+2j, 3-4j], dtype="complex_float32")

# Create a bit-packed boolean vector (1 bit per element)
v3 = pc.vector([True, False, True, True])
```

## Typed NumPy Constructors

If you already have a `numpy.ndarray` and want an explicitly-typed PyCauset vector, most concrete vector classes support a direct NumPy constructor.

```python
import numpy as np
import pycauset as pc

v64 = pc.vector(np.array([1.0, 2.0, 3.0], dtype=np.float64), dtype="float64")
v32 = pc.vector(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype="float32")
vi = pc.vector(np.array([1, 2, 3], dtype=np.int32), dtype="int32")
```

**Requirements**:

- The array must be 1D.
- The dtype must match the target class (e.g. `DenseVector` (dtype `float32`) requires `np.float32`).

## Unit Vectors

A `UnitVector` is a specialized vector type representing a standard basis vector (a vector with a single `1` at a specific index and `0` everywhere else). It is highly optimized for storage and arithmetic operations.

```python
import pycauset as pc

# Create a unit vector of size 1000 with the 1 at index 5
u = pc.UnitVector(1000, 5)

print(u[5]) # 1.0
```

Storage is O(1) (the size and the index). Arithmetic with a unit vector writes it out as a
dense float64 vector first, so it costs the same as with a dense vector.

## Vector Arithmetic

Vectors support `+`, `-`, `*`, `/` (element-wise, with another vector of the same length, a
1-D NumPy array, or a scalar, on either side) and unary `-`. A vector behaves exactly like a
1 x n matrix: the same kernels and the same result-dtype rules
(`documentation/internals/DType System.md`), which follow NumPy:

* integer op Python int keeps the integer dtype and wraps like NumPy (`int8` stays `int8`);
  a Python int that doesn't fit raises `OverflowError`;
* integer op Python float gives `float64`; true division of integers gives a float;
* bool/bit vectors count: `bit + bit` and `bit * 3` give `int32` (NumPy keeps `bool` for
  `+`; PyCauset warns once per call site with `PyCausetDTypeWarning`).

Orientation (row/column) is taken from the left operand.

```python
v1 = pc.vector([1, 2, 3])
v2 = pc.vector([4, 5, 6])
v1 + v2      # [5, 7, 9]
v2 - v1      # [3, 3, 3]
v1 * v2      # [4, 10, 18]
v1 / 2       # [0.5, 1.0, 1.5]   (float64)
10 - v1      # [9, 8, 7]
-v1          # [-1, -2, -3]
v1 * 2.0     # [2.0, 4.0, 6.0]  (float64)
```

### Dot Product

You can compute the dot product of two vectors using [[pycauset.dot]] or the `dot` method.

```python
import pycauset as pc

v1 = pc.vector([1, 2, 3])
v2 = pc.vector([4, 5, 6])

result = pc.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32 (an int: integer vectors give an exact int)
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
v = pc.vector([1, 2, 3])
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
v1 = pc.vector([1, 2, 3])
v2 = pc.vector([4, 5, 6])

# Row @ Column -> Scalar
scalar = v1.T @ v2  # 32

# Column @ Column (Numpy behavior for 1D arrays)
scalar = v1 @ v2    # 32
```

**Type Safety**:
*   Integer or bool vectors (any widths) -> an exact Python `int` (128-bit accumulation;
    `OverflowError` beyond that). Reductions never wrap silently.
*   Any float operand -> `float`; any complex operand -> `complex`.

### Outer Product

The outer product produces a matrix.

```python
v1 = pc.vector([1, 2, 3]) # Column
v2 = pc.vector([4, 5, 6]) # Column

# Column @ Row -> Matrix (N x N)
M = v1 @ v2.T 
# M is:
# [[ 4,  5,  6],
#  [ 8, 10, 12],
#  [12, 15, 18]]
```

**Type Safety**:
*   [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `bool`) @ [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `bool`).T -> [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `bool`) (Logical AND)
*   [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `int32`) @ [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `int32`).T -> [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `int32`)
*   Others -> [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `float64`)

### Matrix-Vector Multiplication

You can multiply matrices and vectors.

```python
M = pc.zeros((3, 3), dtype=pc.float64)
v = pc.vector([1, 1, 1])

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
You can create vectors from NumPy arrays using [[docs/functions/pycauset.vector.md|pycauset.vector]]:
```python
arr = np.array([1.0, 2.0, 3.0])
v = pc.vector(arr)
```

### Mixed Operations
You can add, subtract, or multiply vectors with NumPy arrays directly. The result remains a persistent `pycauset` vector.

```python
v = pc.vector([1, 2, 3])
arr = np.array([10, 10, 10])
v_new = v + arr # [11, 12, 13]
```

## Persistence

Like matrices, vectors are backed by storage (RAM or disk). You can save them permanently using [[pycauset.save]].

```python
import pycauset as pc

v = pc.vector([1, 2, 3])
pc.save(v, "my_vector.pycauset")

v_loaded = pc.load("my_vector.pycauset")
```

## Mixed Types

Operations between different vector dtypes (e.g., integer + float) are supported. The result kind follows the fundamental-kind rule from `documentation/internals/DType System.md` (if a float participates, the result kind is float).

```python
v_int = pc.vector([1, 2], dtype=pc.int32)
v_float = pc.vector([0.5, 0.5], dtype=pc.float64)

v_sum = v_int + v_float  # [1.5, 2.5] (DenseVector (float64))
```

