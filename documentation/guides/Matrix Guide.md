The back-bone of pycauset is the matrix system. While most users will interact with the high-level [[pycauset.CausalSet]] class, the matrix engine powers everything underneath. It is built from the ground-up to allow a seamless workflow as similar to possible to numpy.

`pycauset` behaves like NumPy at small scales (storing data in RAM), but converts to a memory-efficient beast at high scales (automatically spilling to disk).

# Creating a Matrix

Matrices can be created using the [[pycauset.Matrix]] factory function. This function is flexible and accepts lists, NumPy arrays, or dimensions. You can also specify the `dtype` to control the underlying storage format.

```python
import pycauset as pc
import numpy as np

# 1. From a list of lists (infers type)
M1 = pc.Matrix([[1, 2], [3, 4]])  # Creates IntegerMatrix

# 2. From a NumPy array
arr = np.random.rand(5, 5)
M2 = pc.Matrix(arr)               # Creates FloatMatrix

# 3. Empty matrix of specific size and type
M3 = pc.Matrix(100, dtype=int)    # 100x100 IntegerMatrix (zeros)
M4 = pc.Matrix(100, dtype=bool)   # 100x100 DenseBitMatrix (zeros)

# 4. Causal Matrix (Specialized Triangular Bit Matrix)
# This is optimized for causal sets (strictly upper triangular)
C = pc.CausalMatrix(100)
```

# Matrix Operations

`pycauset` provides efficient implementations for matrix operations, mirroring `numpy` semantics where appropriate but optimized for the specific matrix structures (e.g., triangular, bit-packed) used in causal analysis.

## Matrix Multiplication (`matmul`)

Matrix multiplication is performed using the [[pycauset.matmul]](A, B) function.
### Supported Types
Currently, matrix multiplication is optimized and exposed for **[[pycauset.TriangularBitMatrix]]**.
### Syntax
```python
import pycauset as pc

# A and B are TriangularBitMatrix instances
C = pc.matmul(A, B)
```

### Return Value
The operation returns an **[[pycauset.IntegerMatrix]]**. 
- Since [[pycauset.TriangularBitMatrix]] entries are boolean (0 or 1), the dot product of a row and a column results in an integer.
- Geometrically, if $A$ and $B$ represent adjacency matrices of graphs, $C_{ij}$ counts the number of paths of length 2 from node $i$ to node $j$ (i.e., number of intermediate nodes $k$ such that $i \to k$ and $k \to j$).
## Element-wise Multiplication

Element-wise multiplication is performed using the standard multiplication operator `*`.
### Syntax
```python
# A and B can be any matrix type (TriangularBitMatrix, TriangularFloatMatrix, etc.)
C = A * B
```
### Semantics
-   $C_{ij} = A_{ij} \times B_{ij}$
-   For [[pycauset.TriangularBitMatrix]], this is equivalent to a bitwise AND operation ($1 \times 1 = 1$, others $0$).
-   Returns a new matrix of the same type as the operands.

## Scalar Multiplication

Scalar multiplication is supported for all matrix types and is highly optimized.
### Syntax
```python
# A is any matrix
B = A * 5.0
C = 0.5 * A
```

# Saving and Storing Matrices
In pycauset, large matrices are automatically stored on your device's storage disk to allow for work with humongous datasets. Small matrices may live in RAM for performance until they grow too large.

### Saving a Matrix
Matrices are backed by temporary files that are deleted when the program exits, unless [[pycauset.keep_temp_files]] is set to `True`. To permanently save a specific matrix, use [[pycauset.save]]. 

**Note:** If you are working with a [[pycauset.CausalSet]], you should use its `.save()` method (or `pycauset.save(causet)`) to save the entire object including metadata. The method below is for raw matrices.

```python
# Save the matrix to a permanent location
pc.save(C, "my_saved_matrix.pycauset")
```

### Loading a Matrix
You can load any previously saved matrix file using [[pycauset.load]]. The function automatically detects the matrix type (Causal, Integer, Float, etc.) from the file header.

```python
# Load a matrix from disk
matrix = pc.load("my_saved_matrix.pycauset")

# Check the type
print(type(matrix)) 
# <class 'pycauset.pycauset.CausalMatrix'> (or IntegerMatrix, etc.)
```

### Temporary Files
By default, `pycauset` manages backing files automatically. Files are stored in a `.pycauset` directory (or `$PYCAUSET_STORAGE_DIR`).
- **Automatic Cleanup**: Temporary files are deleted on exit.
- **Persistence**: Set `pycauset.keep_temp_files = True` to prevent deletion of temporary files (useful for debugging).
- **Explicit Saving**: Use [[pycauset.save]] to keep specific matrices.



# Matrix Hierarchy

All matrix types derive from a shared C++ `MatrixBase` that owns the memory-mapped backing file and lifecycle management. The hierarchy is designed to support both dense and sparse/triangular structures efficiently.

```mermaid
classDiagram
    class MatrixBase {
        +uint64_t n_
        +double scalar_
        +get_element_as_double()
    }
    class DenseMatrix~T~ {
        +get(i, j) T
    }
    class TriangularMatrix~T~ {
        +vector~uint64_t~ row_offsets_
        +get(i, j) T
    }
    
    MatrixBase <|-- DenseMatrix
    MatrixBase <|-- TriangularMatrix
```

### Common Types
| Python Class | C++ Implementation | Description |
| :--- | :--- | :--- |
| `IntegerMatrix` | `DenseMatrix<int32_t>` | Dense matrix of 32-bit integers. |
| `FloatMatrix` | `DenseMatrix<double>` | Dense matrix of 64-bit floats. |
| `DenseBitMatrix` | `DenseMatrix<bool>` | Dense matrix of booleans (bit-packed). |
| `TriangularBitMatrix` | `TriangularMatrix<bool>` | Strictly upper triangular boolean matrix (Causal Matrix). |
| `TriangularFloatMatrix` | `TriangularMatrix<double>` | Strictly upper triangular float matrix. |

For working with causal matrices (a backbone of the causal set theory), `TriangularBitMatrix` is the primary boolean specialization.  It is exposed as `pycauset.CausalMatrix`. `IntegerMatrix` stores 32-bit counts (e.g., from matrix multiplication). `TriangularFloatMatrix` and `FloatMatrix` (dense) provide floating-point storage for analytical results.