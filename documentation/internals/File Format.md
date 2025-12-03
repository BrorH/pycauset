# PyCauset File Format Specification

PyCauset uses a **ZIP-based container format** (`.pycauset`) for storing large numerical objects efficiently on disk. This format replaces the legacy raw binary format (v2) and provides a self-describing, extensible structure while maintaining high-performance memory mapping capabilities.

## File Structure

A `.pycauset` file is a standard ZIP archive containing two primary files:

1.  **`metadata.json`**: A JSON file containing all object metadata (dimensions, type, seed, etc.).
2.  **`data.bin`**: A raw binary file containing the matrix/vector data.

**Crucially**, the `data.bin` file is stored **uncompressed** (`ZIP_STORED`) and is **aligned** to a 4096-byte page boundary within the ZIP archive. This allows the C++ engine to memory-map the data directly from the ZIP file without extraction, achieving zero-copy performance.

```
[ ZIP Archive Structure ]
+-----------------------+
|     ZIP Headers       |
+-----------------------+
|    metadata.json      | (Compressed or Stored)
+-----------------------+
|      Padding          | (To align data.bin)
+-----------------------+
|      data.bin         | (UNCOMPRESSED, Page Aligned)
|  [ Raw Binary Data ]  |
+-----------------------+
|   Central Directory   |
+-----------------------+
```

## Metadata Schema (`metadata.json`)

The metadata file describes the object stored in `data.bin`.

```json
{
  "pycauset_version": "3.0",
  "object_type": "IntegerMatrix",
  "shape": [1000, 1000],
  "dtype": "int32",
  "storage": "dense",
  "seed": 12345,
  "scalar": 1.0,
  "is_transposed": false
}
```

### Fields

*   **`object_type`**: The high-level class name (e.g., `CausalMatrix`, `IntegerMatrix`, `DenseMatrix`, `Vector`).
*   **`shape`**: Array `[rows, cols]`.
*   **`dtype`**: The data type of elements (`bit`, `int32`, `float64`, `complex128`).
*   **`storage`**: The storage layout (`dense`, `triangular`).
*   **`seed`**: Generation seed (integer).
*   **`scalar`**: Scaling factor (float).
*   **`is_transposed`**: Boolean flag indicating if the matrix is logically transposed.

## Causal Set Archive Format (`.causet`)

While standard matrices use the generic schema above, Causal Set objects (`CausalSet`) use a specialized metadata schema to preserve the spacetime manifold and sprinkling parameters used to generate them.

A `.causet` file is also a ZIP archive containing `metadata.json` and `data.bin`, but the `metadata.json` has additional fields.

### Causal Set Metadata Schema

```json
{
  "pycauset_version": "3.0",
  "object_type": "CausalSet",
  "n": 5000,
  "seed": 987654321,
  "spacetime": {
    "type": "MinkowskiDiamond",
    "args": {
      "dimension": 4
    }
  },
  "matrix_type": "CAUSAL",
  "data_type": "BIT",
  "rows": 5000,
  "cols": 5000,
  "scalar": 1.0,
  "is_transposed": false
}
```

### Specific Fields

*   **`object_type`**: Must be `"CausalSet"`.
*   **`n`**: The number of elements in the set.
*   **`spacetime`**: An object describing the manifold.
    *   **`type`**: Class name of the spacetime (e.g., `MinkowskiDiamond`, `MinkowskiCylinder`).
    *   **`args`**: Dictionary of arguments to reconstruct the spacetime (e.g., `dimension`, `height`, `circumference`).
*   **`matrix_type`**: Always `"CAUSAL"` for the underlying adjacency matrix.
*   **`data_type`**: Always `"BIT"`.

The `data.bin` file in a `.causet` archive contains the **Causal Matrix** (Triangular Bit Matrix) representing the causal relations between the sprinkled points.

## Binary Data (`data.bin`)

The `data.bin` file contains the raw binary representation of the matrix elements. It has **no header**.

### Layouts

#### Dense Layout
Elements are stored row-major.
*   **Size**: $N \times M \times \text{sizeof(T)}$ bytes.
*   **Bit Matrices**: Stored as packed bits. Each row is padded to a 64-bit word boundary.

#### Triangular Layout
Strictly upper triangular matrices store only the elements above the diagonal ($j > i$).
*   **Row 0**: Elements $(0, 1) \dots (0, N-1)$
*   **Row 1**: Elements $(1, 2) \dots (1, N-1)$
*   ...
*   **Row N-2**: Element $(N-2, N-1)$

Each row is padded to a 64-bit boundary for alignment.


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
