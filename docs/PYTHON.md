# Causal++ Python Module Documentation

## Overview
The `pycauset` module provides a high-performance interface for working with massive Causal Matrices (up to $N=10^6$). It uses memory-mapped files to handle data larger than RAM.

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
# Create a 1000x1000 matrix backed by 'my_matrix.bin'
# If the file exists, it loads it. If not, it creates it.
C = pycauset.CausalMatrix(1000, "my_matrix.bin")

print(C)
# Output: <CausalMatrix shape=(1000, 1000)>
```

### Accessing Elements
You can use standard Python indexing `[row, col]`.
**Note**: The matrix is strictly upper triangular. $i < j$ is required.
```python
# Set a causal link: 0 -> 5
C[0, 5] = True

# Check a link
exists = C[0, 5]  # True
```

### Matrix Multiplication
Use the `@` operator for matrix multiplication. This calculates the number of paths of length 2 between nodes.
```python
# A -> B -> C implies A -> C
A = pycauset.CausalMatrix(100, "A.bin")
A[0, 10] = True
A[10, 20] = True

# Result is an IntegerMatrix (counts paths)
# Note: This creates a temporary file for the result automatically.
Result = A @ A 

print(Result[0, 20]) 
# Output: 1 (There is 1 path: 0->10->20)
```

### Explicit Multiplication (Control Output File)
If you want to specify where the result is stored (recommended for large matrices):
```python
# Result will be stored in 'paths.bin'
Result = A.multiply(A, result_file="paths.bin")
```

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
