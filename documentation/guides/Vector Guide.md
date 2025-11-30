# Vector Guide

`pycauset` introduces efficient, disk-backed vectors that integrate seamlessly with the matrix operations.

## Creating Vectors

You can create vectors using the `Vector` factory function. It automatically selects the most efficient storage backend based on your data.

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

You can compute the dot product of two vectors using `pycauset.dot` or the `dot` method.

```python
from pycauset import dot

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

result = dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32.0
# OR
result = v1.dot(v2)
```

## Persistence

Like matrices, vectors are backed by files on disk. You can save them permanently using `pycauset.save`.

```python
from pycauset import save, load

v = Vector([1, 2, 3])
save(v, "my_vector.pycauset")

v_loaded = load("my_vector.pycauset")
```

## Mixed Types

Operations between different vector types (e.g., IntegerVector + FloatVector) are supported. The result is typically promoted to a `FloatVector` (DenseVector<double>) to ensure precision.

```python
v_int = Vector([1, 2], dtype="int")
v_float = Vector([0.5, 0.5], dtype="float")

v_sum = v_int + v_float  # [1.5, 2.5] (FloatVector)
```
