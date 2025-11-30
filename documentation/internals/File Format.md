# PyCauset File Format Specification

PyCauset uses a custom binary format for storing large numerical objects (Matrices, Vectors) efficiently on disk. The format is designed to support memory mapping, allowing datasets larger than available RAM to be processed with high performance.

## File Structure

Each file consists of a **4096-byte header** followed immediately by the raw binary data.

```
+-----------------------+
|      File Header      |  4096 Bytes (Page Aligned)
+-----------------------+
|                       |
|      Binary Data      |  Variable Size
|                       |
+-----------------------+
```

### Why 4096 Bytes?
The header size is explicitly set to 4096 bytes to match the standard **Memory Page Size** of most modern Operating Systems (Windows, Linux, macOS). 

*   **Page Alignment**: By reserving exactly one page for the header, the actual data payload starts at the beginning of the second page (Offset 4096). This ensures that the data is **page-aligned**, which is critical for performance.
*   **Vectorization**: Aligned data allows the CPU to use SIMD (Single Instruction, Multiple Data) instructions (like AVX2/AVX-512) more effectively.
*   **Future Proofing**: The large header provides ample space for future metadata (e.g., coordinate systems, labels, JSON descriptions) without needing to shift the massive binary data that follows.

## File Header

The header contains metadata necessary to identify the object type, dimensions, and data format. It is defined as follows:

| Offset | Size       | Type      | Description                |
| :----- | :--------- | :-------- | :------------------------- |
| 0      | 8 bytes    | char[8]   | Magic Number: `"PYCAUSET"` |
| 8      | 4 bytes    | uint32_t  | Pycauset version (2)       |
| 12     | 4 bytes    | uint32_t  | Matrix Type (Enum)         |
| 16     | 4 bytes    | uint32_t  | Data Type (Enum)           |
| 20     | 8 bytes    | uint64_t  | Rows ($N$)                 |
| 28     | 8 bytes    | uint64_t  | Columns ($M$)              |
| 36     | 8 bytes    | uint64_t  | Seed (0 if not applicable) |
| 44     | 8 bytes    | double    | Scalar (Scaling factor)    |
| 52     | 1 byte     | uint8_t   | Is Temporary (1=True, 0=False) |
| 53     | 1 byte     | uint8_t   | Is Transposed (1=True, 0=False) |
| 54     | 4042 bytes | uint8_t[] | Reserved / Padding         |

### Enums

**Matrix Type (`uint32_t`)**
*   `1`: **CAUSAL** (Strictly Upper Triangular Matrix, Boolean/Bit storage)
*   `2`: **INTEGER** (Strictly Upper Triangular Matrix, 32-bit Integer storage)
*   `3`: **TRIANGULAR_FLOAT** (Strictly Upper Triangular Matrix, 64-bit Float storage)
*   `4`: **DENSE_FLOAT** (Dense $N \times M$ Matrix)
*   `5`: **IDENTITY** (Virtual Identity Matrix, no data on disk)
*   `6`: **VECTOR** (Dense Vector, $N$ elements)

**Data Type (`uint32_t`)**
*   `1`: **BIT** (1 bit per element)
*   `2`: **INT32** (32-bit signed integer)
*   `3`: **FLOAT64** (64-bit double precision float)
*   `4`: **COMPLEX_FLOAT64** (128-bit complex number: 2x double)

## Storage Strategies

### 1. Causal / Triangular Matrices
*   **Logical Structure**: Strictly Upper Triangular ($i < j$).
*   **Storage**: Row-major. Row $i$ stores columns $j = i+1 \dots N-1$.
*   **Bit-Packing**: If `DataType` is `BIT`, rows are padded to align to 64-bit boundaries.
*   **Size Calculation**: $\sum_{i=0}^{N-1} \lceil \frac{N - 1 - i}{64} \rceil \times 8$ bytes.

### 2. Dense Matrices
*   **Logical Structure**: Standard $N \times M$ matrix.
*   **Storage**: Row-major (C-style).
*   **Size**: $N \times M \times \text{sizeof(DataType)}$.

### 3. Vectors
*   **Logical Structure**: 1D Array of size $N$.
*   **Storage**: Contiguous block of memory.
*   **Orientation**: The `Is Transposed` flag in the header determines if it is treated as a Column Vector (default) or Row Vector.
*   **Size**: $N \times \text{sizeof(DataType)}$.

### 4. Identity Matrix
*   **Logical Structure**: Diagonal matrix where $M_{ii} = 1$.
*   **Storage**: **None**. The file consists *only* of the 4096-byte header. The values are generated virtually at runtime.

## Loading

The `pycauset.load(path)` function reads the header to determine the object type and instantiates the appropriate Python class (`CausalMatrix`, `IntegerMatrix`, `Vector`, etc.), wrapping the memory-mapped data.
