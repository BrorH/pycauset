
`pycauset` provides efficient implementations for matrix operations, mirroring `numpy` semantics where appropriate but optimized for the specific matrix structures (e.g., triangular, bit-packed) used in causal analysis.

## Matrix Multiplication (`matmul`)

Matrix multiplication is performed using the `pycauset.matmul(A, B)` function.

### Supported Types
Matrix multiplication is supported for all combinations of matrix types. The return type depends on the operands:

| Operand A | Operand B | Result Type |
| :--- | :--- | :--- |
| `FloatMatrix` (Dense) | Any | `FloatMatrix` |
| Any | `FloatMatrix` (Dense) | `FloatMatrix` |
| `TriangularFloatMatrix` | Triangular (Any) | `TriangularFloatMatrix` |
| Triangular (Any) | `TriangularFloatMatrix` | `TriangularFloatMatrix` |
| `IntegerMatrix` | `IntegerMatrix` or `TriangularBitMatrix` | `IntegerMatrix` |
| `TriangularBitMatrix` | `IntegerMatrix` | `IntegerMatrix` |
| `TriangularBitMatrix` | `TriangularBitMatrix` | `IntegerMatrix` |
| `DenseBitMatrix` | `DenseBitMatrix` | `IntegerMatrix` |

**Note on `IntegerMatrix`**: In `pycauset`, `IntegerMatrix` is a **dense** matrix storing 32-bit integers. It is the standard return type for discrete matrix multiplications (like path counting).

### Syntax
```python
import pycauset

# A and B can be of any supported type
C = pycauset.matmul(A, B)
```

### Return Value
See the table above. The system automatically promotes types to the most general required structure (Dense > TriangularFloat > Integer > Bit).

### Implementation Details
The implementation uses memory-mapped files to handle large matrices without loading them entirely into RAM.
- **Triangular Matrices**: Uses a row-addition algorithm that exploits the strictly upper triangular structure. For each row $i$ of $A$, it accumulates rows $k$ of $B$ where $A_{ik} \neq 0$.
- **Dense Matrices**: Uses a row-wise accumulation (IKJ) algorithm optimized for row-major storage.
- **Mixed Types**: When multiplying mixed types (e.g., `TriangularBitMatrix` * `FloatMatrix`), the operation is optimized to use the sparse structure of the triangular matrix while producing a dense result, avoiding full expansion of the sparse matrix before multiplication.
- **Scalar Multiplication**: Scalar factors are handled lazily and are correctly propagated during matrix multiplication ($C.scalar = A.scalar \times B.scalar$).

This approach ensures that operations are performed chunk-wise (row-wise), respecting memory constraints.

## Element-wise Multiplication

Element-wise multiplication is performed using the standard multiplication operator `*`.

### Syntax
```python
# A and B can be any matrix type (TriangularBitMatrix, TriangularFloatMatrix, etc.)
C = A * B
```

### Semantics
-   $C_{ij} = A_{ij} \times B_{ij}$
-   For `TriangularBitMatrix`, this is equivalent to a bitwise AND operation ($1 \times 1 = 1$, others $0$).
-   Returns a new matrix of the same type as the operands.

## Scalar Multiplication

Scalar multiplication is supported for all matrix types and is highly optimized.

### Syntax
```python
# A is any matrix
B = A * 5.0
C = 0.5 * A
```

### Lazy Evaluation
`pycauset` uses **lazy evaluation** for scalar multiplication:
-   The operation is $O(1)$.
-   It does **not** iterate over the matrix data.
-   Instead, it updates an internal `scalar` field in the matrix object.
-   When elements are accessed (e.g., via `get_element_as_double` or converted to numpy), the stored value is multiplied by this scalar on the fly.
-   If you perform `C = pycauset.matmul(A, B)`, the resulting matrix will inherit the product of the scalars: `C.scalar = A.scalar * B.scalar`.