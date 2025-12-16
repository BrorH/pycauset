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

### `get(i: int, j: int) -> bool`
Get the value at row `i` and column `j`.

### `set(i: int, j: int, value: bool)`
Set the value at row `i` and column `j`.

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

### `__getitem__(idx: tuple) -> bool`
Get element using `[i, j]` syntax.

### `__setitem__(idx: tuple, value: bool)`
Set element using `[i, j] = value` syntax.

### `__repr__() -> str`
String representation of the matrix.
