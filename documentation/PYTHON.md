# Causal++ Python Module Documentation

## Overview
The `pycauset` module provides a high-performance interface for working with massive Causal Matrices (up to $N=10^6$). It uses memory-mapped files to handle data larger than RAM.

### Matrix hierarchy
All matrix types derive from a shared C++ `MatrixBase` that owns the memory-mapped backing file, row-offset bookkeeping, and lifecycle management. `CausalMatrix` (exported as `pycauset.CausalMatrix`) is the primary boolean specialization exposed to Python today; multiplication via `pycauset.matmul` produces an `IntegerMatrix`, another `MatrixBase` subclass that stores 32-bit counts. This layered design mirrors NumPy’s array family and allows future matrix flavors (e.g., diagonal-enabled variants) to hook into the same storage and cleanup guarantees without changing the Python surface area.

## Installation
Ensure the module is built and copied into `python/pycauset/` by running:
```powershell
./build.ps1 -Python

# Run Python scripts with the interpreter version used during the build
py -3.12 your_script.py
```

## Usage

### Importing
```python
import pycauset
```

### Creating a Matrix
```python
# Create a 1000x1000 matrix backed by 'my_matrix.pycauset'.
# Simple names are stored inside ./.pycauset by default.
C = pycauset.CausalMatrix(1000, "my_matrix")

print(C)
# Output:
# CausalMatrix(shape=(1000, 1000))
# [
#  [0 0 0 ... 0 0 1]
#  ...
# ]
```

### Loading a Matrix
You can load any previously saved matrix file using `pycauset.load()`. The function automatically detects the matrix type (Causal, Integer, Float, etc.) from the file header.

```python
# Load a matrix from disk
matrix = pycauset.load("my_matrix.pycauset")

# Check the type
print(type(matrix)) 
# <class 'pycauset.pycauset.CausalMatrix'> (or IntegerMatrix, etc.)

# Use it as normal
print(matrix.size())
```

### Convenience Creator
`pycauset.Matrix(...)` is an in-memory helper for quick experiments. Passing an integer mirrors the `CausalMatrix` constructor (minus the file-backed storage); nested Python sequences, NumPy arrays, or any object exposing `.size()`/`.get(i, j)` are accepted as-is—even when the data are dense or contain arbitrary numeric values.

```python
# Integer input behaves like the CausalMatrix constructor
alpha = pycauset.Matrix(256, populate=False)

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

Providing `saveas=` now only issues a warning because dense `Matrix` instances never persist to disk; use `pycauset.CausalMatrix` whenever you need the memory-mapped, upper-triangular storage that backs the C++ engine.

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
Constructors default to `populate=True`. `pycauset.CausalMatrix` fills the strict upper triangle with Bernoulli(0.5) draws while the in-memory `pycauset.Matrix` populates every entry. Control randomness via either:

- The module-level toggle: `pycauset.seed = 42` makes every subsequent populated matrix deterministic until you reset it to `None`.
- Per-call overrides: pass `seed=42` to `pycauset.CausalMatrix(...)` or `pycauset.CausalMatrix.random(...)`.

Set `populate=False` whenever you want an empty matrix, or stick with `pycauset.Matrix([...])` to populate from explicit data.

### Elementwise vs Matrix Multiplication
`*` now mirrors NumPy’s elementwise semantics: it returns a new `CausalMatrix` whose entries are the logical AND of the operands. Use this when you want to intersect adjacency structures without leaving the boolean domain.

```python
lhs = pycauset.CausalMatrix(100)
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
