# PyCauset File Format Specification

PyCauset uses a custom binary format for storing large matrices efficiently on disk. The format is designed to support memory mapping, allowing matrices larger than available RAM to be processed.

## File Structure

Each file consists of a **4096-byte header** followed immediately by the raw matrix data.

```
+-----------------------+
|      File Header      |  4096 Bytes
+-----------------------+
|                       |
|      Matrix Data      |  Variable Size
|                       |
+-----------------------+
```

## File Header

The header contains metadata necessary to identify the file type, matrix dimensions, and data format. It is defined as follows:

| Offset | Size       | Type      | Description                |
| :----- | :--------- | :-------- | :------------------------- |
| 0      | 8 bytes    | char[8]   | Magic Number: `"PYCAUSET"` |
| 8      | 4 bytes    | uint32_t  | Pycauset version (2)       |
| 12     | 4 bytes    | uint32_t  | Matrix Type (Enum)         |
| 16     | 4 bytes    | uint32_t  | Data Type (Enum)           |
| 20     | 4 bytes    | -         | Padding (Alignment)        |
| 24     | 8 bytes    | uint64_t  | Rows ($N$)                 |
| 32     | 8 bytes    | uint64_t  | Columns ($N$)              |
| 40     | 8 bytes    | uint64_t  | Seed (0 if not applicable) |
| 48     | 8 bytes    | double    | Scalar (Scaling factor)    |
| 56     | 1 byte     | uint8_t   | Is Temporary (1=True, 0=False) |
| 57     | 4039 bytes | uint8_t[] | Reserved / Padding         |

### Enums

**Matrix Type (`uint32_t`)**
*   `1`: **CAUSAL** (Strictly Upper Triangular, Boolean/Bit storage)
*   `2`: **INTEGER** (Strictly Upper Triangular, 32-bit Integer storage)
*   `3`: **TRIANGULAR_FLOAT** (Strictly Upper Triangular, 64-bit Float storage)
*   `4`: **DENSE_FLOAT** (Dense $N \times N$, 64-bit Float storage)

**Data Type (`uint32_t`)**
*   `1`: **BIT** (1 bit per element)
*   `2`: **INT32** (32-bit signed integer)
*   `3`: **FLOAT64** (64-bit double precision float)

## Storage Strategies

### Causal Matrix (Type 1)
*   **Logical Structure**: Strictly Upper Triangular ($i < j$).
*   **Storage**: Bit-packed.
*   **Layout**: Row-major. Each row $i$ stores elements for columns $j = i+1 \dots N-1$.
*   **Alignment**: Each row is padded to align to a 64-bit boundary.
*   **Size Calculation**: $\sum_{i=0}^{N-1} \lceil \frac{N - 1 - i}{64} \rceil \times 8$ bytes.

### Integer Matrix (Type 2)
*   **Logical Structure**: Strictly Upper Triangular ($i < j$).
*   **Storage**: `int32_t` (4 bytes).
*   **Layout**: Row-major. Row $i$ stores columns $j = i+1 \dots N-1$.
*   **Size Calculation**: $\frac{N(N-1)}{2} \times 4$ bytes.

### Triangular Float Matrix (Type 3)
*   **Logical Structure**: Strictly Upper Triangular ($i < j$).
*   **Storage**: `double` (8 bytes).
*   **Layout**: Row-major. Row $i$ stores columns $j = i+1 \dots N-1$.
*   **Size Calculation**: $\frac{N(N-1)}{2} \times 8$ bytes.

### Dense Float Matrix (Type 4)
*   **Logical Structure**: Dense ($N \times N$).
*   **Storage**: `double` (8 bytes).
*   **Layout**: Standard Row-major.
*   **Size Calculation**: $N^2 \times 8$ bytes.

## Loading

The `pycauset.load(path)` function reads the header to determine the matrix type and instantiates the appropriate Python class (`CausalMatrix`, `IntegerMatrix`, etc.), wrapping the memory-mapped data.
