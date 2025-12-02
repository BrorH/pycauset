# PyCauset File Format Specification

PyCauset uses a custom binary format (`.pycauset`) for storing various large numerical objects efficiently on disk. While originally designed for matrices, the format now supports multiple object types including **Dense Matrices**, **Triangular Matrices**, **Bit-Packed Matrices**, and **Vectors**. The format is designed to support memory mapping, allowing datasets larger than available RAM to be processed with high performance.

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
| 12     | 4 bytes    | uint32_t  | Object Type (Enum)         |
| 16     | 4 bytes    | uint32_t  | Data Type (Enum)           |
| 20     | 8 bytes    | uint64_t  | Rows ($N$)                 |
| 28     | 8 bytes    | uint64_t  | Columns ($M$)              |
| 36     | 8 bytes    | uint64_t  | Seed (0 if not applicable) |
| 44     | 8 bytes    | double    | Scalar (Scaling factor)    |
| 52     | 1 byte     | uint8_t   | Is Temporary (1=True, 0=False) |
| 53     | 1 byte     | uint8_t   | Is Transposed (1=True, 0=False) |
| 54     | 4042 bytes | uint8_t[] | Reserved / Padding         |

### Enums

**Object Type (`uint32_t`)**

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

# Causal Set Archive Format (.causet)

While the `.pycauset` format is optimized for raw numerical data, a **Causal Set** is a higher-level object that combines a causal matrix with spacetime metadata (dimension, shape, coordinates, etc.).

To store this efficiently and portably, PyCauset uses a **ZIP-based archive format** with the extension `.causet`.

## Archive Structure

A `.causet` file is a standard ZIP archive containing two specific files:

```
my_universe.causet (ZIP Archive)
├── metadata.json    (JSON Object)
└── matrix.bin       (Binary .pycauset file)
```

### 1. metadata.json

This file contains all the parameters required to reconstruct the `CausalSpacetime` object that generated the set.

**Example:**
```json
{
    "dim": 4,
    "shape": "cylinder",
    "height": 1.5,
    "radius": 1.0,
    "size": 1000,
    "coordinates": [[0.1, 0.2, ...], ...],  // Optional: if coordinates are saved
    "version": "1.0"
}
```

### 2. matrix.bin

This is the raw adjacency matrix of the causal set, stored in the standard **PyCauset Binary Format** (described above). It is usually a `CausalMatrix` (Strictly Upper Triangular Bit Matrix).

## Loading Process

When `pycauset.load("my_universe.causet")` is called:

1.  PyCauset detects the `.causet` extension and treats it as a ZIP archive.
2.  It extracts `metadata.json` to read the spacetime parameters.
3.  It extracts `matrix.bin` to a temporary location (or memory maps it directly if supported).
4.  It reconstructs the `CausalSpacetime` object (e.g., `MinkowskiCylinder`) using the metadata.
5.  It loads the matrix using the binary loader.
6.  It returns a `CausalSet` object linking the spacetime and the matrix.
