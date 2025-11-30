# Pycauset User Guide

## Overview
The `pycauset` module provides a high-performance interface for working with massive Causal Matrices (up to $N=10^6$). It uses memory-mapped files to handle data larger than RAM.

## Usage

### Importing
```python
import pycauset
```

### Creating a Matrix
The fundamental class in `pycauset` is `pycauset.Matrix`. You can think of it as similar to a NumPy array, but restricted to square matrices.

#### `pycauset.Matrix`
This is the general-purpose matrix factory. It can be created from lists, NumPy arrays, or by specifying a size. You can optionally specify a `dtype` to force a specific internal representation.

```python
# Create from a list of lists (infers type)
data = [
    [1, 2],
    [3, 4]
]
M = pycauset.Matrix(data) # Returns IntegerMatrix

# Create an empty (zero-filled) 5x5 integer matrix
M_int = pycauset.Matrix(5, dtype=int)

# Create a dense boolean matrix
M_bool = pycauset.Matrix(1000, dtype=bool)
```

#### `pycauset.CausalMatrix`
For causal set theory, we often deal with causal matrices that are binary (0 or 1) and strictly upper triangular (representing the causal order). The `CausalMatrix` function is the standard way to create these. It returns a highly optimized `TriangularBitMatrix`.

```python
# Create a 1000x1000 causal matrix
# By default, it populates with random Bernoulli(0.5) data
C = pycauset.CausalMatrix(1000)

# Create from a list (automatically converts to binary and enforces triangularity)
data = [
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0]
]
C_from_list = pycauset.CausalMatrix(data)
```

### Creating a Vector
`pycauset` also supports efficient, disk-backed vectors. See the [Vector Guide](Vector%20Guide.md) for details.

```python
# Create a vector from a list
v = pycauset.Vector([1, 2, 3])

# Create a large empty vector
v_large = pycauset.Vector(1000000, dtype="float")
```

### Matrix Operations
```

#### Other Matrix Types
The `Matrix` factory returns specialized classes based on the data:
*   **`pycauset.IntegerMatrix`**: Dense $N \times N$ matrix storing 32-bit integers.
*   **`pycauset.TriangularIntegerMatrix`**: Strictly upper triangular matrix storing 32-bit integers.
*   **`pycauset.FloatMatrix`**: Dense $N \times N$ matrix storing 64-bit doubles.
*   **`pycauset.TriangularFloatMatrix`**: Strictly upper triangular matrix storing 64-bit doubles.
*   **`pycauset.DenseBitMatrix`**: Dense $N \times N$ matrix storing boolean values (1 bit per element).
*   **`pycauset.TriangularBitMatrix`**: Strictly upper triangular matrix storing boolean values (1 bit per element).

### Matrix Operations

#### Multiplication
```python
A = pycauset.CausalMatrix(100)
B = pycauset.CausalMatrix(100)

# Matrix multiplication (A @ B)
# Returns a TriangularIntegerMatrix counting paths of length 2
AB = pycauset.matmul(A, B)
```

#### The $K$ Matrix
Computes $K = C(aI + C)^{-1}$, a key quantity in causal set theory.
```python
# Returns a TriangularFloatMatrix
K = pycauset.compute_k(C, a=1.0)
```

#### Lazy Scaling
All matrices support $O(1)$ scalar multiplication. The actual values are scaled on-the-fly when accessed via `get_element_as_double` (or implicitly when converting to other formats).

```python
# Multiply matrix by scalar (instantaneous)
K_scaled = K * 2.5 
```

### Saving a Matrix
Matrices are backed by temporary files that are deleted when the program exits, unless `pycauset.keep_temp_files` is set to `True`. To permanently save a matrix, use `pycauset.save()`. This creates a hard link to the backing file if possible, avoiding data duplication.

```python
# Save the matrix to a permanent location
pycauset.save(C, "my_saved_matrix.pycauset")
```

### Loading a Matrix
You can load any previously saved matrix file using `pycauset.load()`. The function automatically detects the matrix type (Causal, Integer, Float, etc.) from the file header.

```python
# Load a matrix from disk
matrix = pycauset.load("my_saved_matrix.pycauset")

# Check the type
print(type(matrix)) 
# <class 'pycauset.pycauset.CausalMatrix'> (or IntegerMatrix, etc.)
```

### Temporary Files
By default, `pycauset` manages backing files automatically. Files are stored in a `.pycauset` directory (or `$PYCAUSET_STORAGE_DIR`).
- **Automatic Cleanup**: Temporary files are deleted on exit.
- **Persistence**: Set `pycauset.keep_temp_files = True` to prevent deletion of temporary files (useful for debugging).
- **Explicit Saving**: Use `pycauset.save()` to keep specific matrices.

### Deprecated Features
- The `saveas` argument in constructors and functions is deprecated. Use `pycauset.save()` instead.

### Convenience Creator
`pycauset.Matrix(...)` is an in-memory helper for quick experiments. Passing an integer mirrors the `CausalMatrix` constructor (minus the file-backed storage); nested Python sequences, NumPy arrays, or any object exposing `.size()`/`.get(i, j)` are accepted as-is—even when the data are dense or contain arbitrary numeric values.

```python
# Integer input behaves like the CausalMatrix constructor
alpha = pycauset.Matrix(256)

# Nested sequences can now describe any square matrix
beta = pycauset.Matrix([
	[1, 0, 0],
	[2, 3, 4],
	[5, 6, 7],
])

# NumPy arrays (or anything convertible via np.asarray) are accepted directly
import numpy as np
gamma = pycauset.Matrix(np.random.rand(4, 4))
```


### Accessing Elements
You can use standard Python indexing `[row, col]`.
```python
# Set a causal link: 0 -> 5
C[0, 5] = True

# Check a link
exists = C[0, 5]  # True
```

Call `str(C)` (or just `print(C)`) to view a NumPy-style preview. Matrices with up to six rows/columns render in full; larger matrices show the first and last three rows/columns with ellipses (`...`) in between.

### Random Population & Seeding
To create a random matrix, use the `.random()` factory method. `pycauset.CausalMatrix.random(N)` fills the strict upper triangle with Bernoulli(0.5) draws.

Control randomness via either:
- The module-level toggle: `pycauset.seed = 42` makes every subsequent random matrix deterministic until you reset it to `None`.
- Per-call overrides: pass `seed=42` to `pycauset.CausalMatrix.random(...)`.

The default constructor `pycauset.CausalMatrix(N)` creates an empty matrix (all zeros).

### Elementwise vs Matrix Multiplication
`*` now mirrors NumPy’s elementwise semantics: it returns a new `CausalMatrix` whose entries are the logical AND of the operands. Use this when you want to intersect adjacency structures without leaving the boolean domain.

```python
lhs = pycauset.CausalMatrix(100)
rhs = pycauset.CausalMatrix(100)
# ... fill both ...
overlap = lhs * rhs  # still a CausalMatrix
```

Use `pycauset.matmul(lhs, rhs)` for true matrix multiplication (`numpy.matmul` semantics). The result is an `IntegerMatrix` containing path counts / integer dot products.

```python
# A -> B -> C implies A -> C
A = pycauset.CausalMatrix(100, "A.bin")
A[0, 10] = True
A[10, 20] = True

Result = pycauset.matmul(A, A)
print(Result[0, 20])  # 1 path: 0->10->20
```

### Explicit Multiplication (Control Output File)
If you want to specify where the result is stored (recommended for large matrices):
```python
# Result will be stored in 'paths.bin'
Result = pycauset.matmul(A, A, saveas="paths.bin")
# Existing code that calls A.multiply(...) still works; matmul is just the numpy-style entry point.

```

### Storage Lifecycle & Cleanup
- When no `backing_file` is provided, matrices are written to `<cwd>/.pycauset/<variable>.pycauset`. The file name is inferred from the assignment target (`alpha = pycauset.causalmatrix(...) -> alpha.pycauset`).
- Set the `PYCAUSET_STORAGE_DIR` environment variable to relocate that hidden directory. Explicit `Path` objects are also respected.
- Auto-generated `.pycauset` files are deleted when the interpreter exits unless you opt in to persistence via `pycauset.save = True` (the default is `False`).
- User-specified paths (anything passed to `backing_file`) are never deleted automatically, though creating a new matrix with the same path will overwrite the data.
- Files left over from a prior `pycauset.save = True` run are removed the next time you exit with `pycauset.save = False`.
- Call `matrix.close()` when you are done with an explicit backing file (or before a temporary directory is torn down) to release the memory-mapped handle immediately.

### IntegerMatrix
The result of a multiplication is an `IntegerMatrix`. It is read-only.
```python
count = Result[0, 20]
shape = Result.shape
```

## Performance Tips
1. **Avoid Loops**: Do not iterate over the matrix in Python (e.g., `for i in range(N)`). This is slow. Use the C++ operations.
2. **Storage**: Ensure you have enough disk space. A $10^6 \times 10^6$ matrix requires ~64GB.
3. **Memory**: The module uses very little RAM, but relies on the OS page cache.
rhs = pycauset.CausalMatrix(100)
# ... populate both ...
overlap = lhs * rhs  # still a CausalMatrix
```

Use `pycauset.matmul(lhs, rhs)` for true matrix multiplication (`numpy.matmul` semantics). The result is an `IntegerMatrix` containing path counts / integer dot products.

```python
# A -> B -> C implies A -> C
A = pycauset.CausalMatrix(100, "A.bin")
A[0, 10] = True
A[10, 20] = True

Result = pycauset.matmul(A, A)
print(Result[0, 20])  # 1 path: 0->10->20
```

### Explicit Multiplication (Control Output File)
If you want to specify where the result is stored (recommended for large matrices):
```python
# Result will be stored in 'paths.bin'
Result = pycauset.matmul(A, A, saveas="paths.bin")
# Existing code that calls A.multiply(...) still works; matmul is just the numpy-style entry point.

```

### Storage Lifecycle & Cleanup
- When no `backing_file` is provided, matrices are written to `<cwd>/.pycauset/<variable>.pycauset`. The file name is inferred from the assignment target (`alpha = pycauset.causalmatrix(...) -> alpha.pycauset`).
- Set the `PYCAUSET_STORAGE_DIR` environment variable to relocate that hidden directory. Explicit `Path` objects are also respected.
- Auto-generated `.pycauset` files are deleted when the interpreter exits unless you opt in to persistence via `pycauset.save = True` (the default is `False`).
- User-specified paths (anything passed to `backing_file`) are never deleted automatically, though creating a new matrix with the same path will overwrite the data.
- Files left over from a prior `pycauset.save = True` run are removed the next time you exit with `pycauset.save = False`.
- Call `matrix.close()` when you are done with an explicit backing file (or before a temporary directory is torn down) to release the memory-mapped handle immediately.

### IntegerMatrix
The result of a multiplication is an `IntegerMatrix`. It is read-only.
```python
count = Result[0, 20]
shape = Result.shape
```

## Performance Tips
1. **Avoid Loops**: Do not iterate over the matrix in Python (e.g., `for i in range(N)`). This is slow. Use the C++ operations.
2. **Storage**: Ensure you have enough disk space. A $10^6 \times 10^6$ matrix requires ~64GB.
3. **Memory**: The module uses very little RAM, but relies on the OS page cache.
