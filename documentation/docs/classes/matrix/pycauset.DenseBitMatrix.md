# pycauset.DenseBitMatrix

A memory-mapped dense matrix storing boolean values (bits). Inherits from [MatrixBase](pycauset.MatrixBase.md).

## Constructor

```python
pycauset.DenseBitMatrix(n: int)
```

## Static Methods

### `random(n: int, density: float, seed: int = None) -> DenseBitMatrix`
Create a random dense bit matrix.

## Properties

### `shape`
Returns a tuple `(rows, cols)` representing the dimensions of the matrix.

## Methods

### Indexing

Element access uses NumPy-style indexing:

```python
x = M[i, j]
M[i, j] = value
```

### `multiply(other: DenseBitMatrix) -> IntegerMatrix`
Multiply this matrix by another `DenseBitMatrix`.

**Returns:**
*   `IntegerMatrix`: The result of the multiplication. Note that this performs integer matrix multiplication (counting paths), not boolean multiplication. The result at $(i, j)$ is the number of paths of length 1 from $i$ to $j$ (which is just the dot product).

**GPU Acceleration:**
This operation is **GPU-accelerated** if a compatible NVIDIA GPU is detected. The implementation uses a highly optimized bit-packed kernel that performs 64 operations per cycle per thread.

**CPU Fallback:**
If no GPU is available, the operation uses optimized AVX-512/NEON `popcount` instructions on the CPU, providing significant speedups over standard loops.


### `__invert__() -> DenseBitMatrix`
Compute the bitwise NOT of the matrix.


### `__repr__() -> str`
String representation of the matrix.
