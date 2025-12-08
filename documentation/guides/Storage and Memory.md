# Storage and Memory Management

PyCauset is designed to handle causal sets and matrices of any size, from small test cases to massive simulations that exceed physical RAM. It achieves this through a **Tiered Storage Architecture**.

## The Philosophy: Tiered Storage

In standard Python (e.g., NumPy), creating a matrix allocates memory in RAM. If you run out of RAM, your program crashes.

In PyCauset, we treat storage as a hierarchy:

1.  **L1 (RAM)**: Small matrices and frequently accessed data live here for maximum speed.
2.  **L2 (Disk)**: Large matrices automatically spill to **memory-mapped files** on your SSD/HDD.

The **Memory Governor** manages this automatically. It monitors your system's available RAM and decides where to place each new object.
*   **Instant Access**: Whether in RAM or on Disk, the API is identical.
*   **Automatic Caching**: For disk-backed objects, the OS automatically keeps frequently used parts in RAM.
*   **Persistence**: Disk-backed objects are persistent. RAM objects are transient but can be saved easily.

## The File Format (`.pycauset`)

PyCauset uses a unified, portable file format for all objects (`CausalSet`, `FloatMatrix`, `IntegerMatrix`, etc.).

A `.pycauset` file is simply a **ZIP Archive** containing two files:

1.  **`metadata.json`**: A human-readable JSON file describing the object (dimensions, data type, generation parameters).
2.  **`data.bin`**: The raw, uncompressed binary data.

> **Note**: Because `data.bin` is stored uncompressed and aligned within the ZIP file, PyCauset can memory-map the data *directly* from the archive without extracting it. This means opening a 100 GB file takes milliseconds.

### Metadata Example

If you unzip a `.pycauset` file, you can inspect `metadata.json`:

```json
{
  "rows": 1000,
  "cols": 1000,
  "seed": 12345,
  "scalar": 1.0,
  "is_transposed": false,
  "matrix_type": "INTEGER",
  "data_type": "INT32"
}
```

### Metadata Fields Explained

*   **`matrix_type`**: The logical mathematical structure (`INTEGER`, `DENSE_FLOAT`, `CAUSAL`, `TRIANGULAR_FLOAT`).
*   **`data_type`**: The underlying binary format of elements (`INT32`, `FLOAT64`, `BIT`).
*   **`rows`**: Number of rows.
*   **`cols`**: Number of columns.
*   **`seed`**: The random seed used to generate the object (if applicable). This allows for reproducibility.
*   **`scalar`**: A global scaling factor applied to all elements (see "Lazy Evaluation" below).
*   **`is_transposed`**: A boolean flag indicating if the matrix is logically transposed (see "Lazy Evaluation" below).

## Compute-Once Caching

PyCauset implements a "compute-once" philosophy for expensive mathematical operations.

### Scalar Properties (Trace, Determinant, Eigenvalues)
When you compute properties like `trace()`, `determinant()`, or `eigenvalues()`, the result is stored in the matrix object's memory.
*   **Automatic Persistence**: When you call `pycauset.save(matrix, path)`, these cached values are written into the `metadata.json` file.
*   **Automatic Restoration**: When you `load()` the matrix later, these values are read from metadata, making them available instantly without recomputation.

### Large Objects (Eigenvectors, Inverse)
For large results that are matrices themselves (like Eigenvectors or the Inverse matrix), PyCauset offers optional persistence to avoid bloating the file unless requested.

```python
# Compute eigenvectors and SAVE them to the archive
vecs = matrix.eigenvectors(save=True)
```

When `save=True` is passed:
1.  The eigenvectors are computed.
2.  The result is written to new binary files (`eigenvectors.real.bin`, `eigenvectors.imag.bin`) **inside** the existing `.pycauset` ZIP archive.
3.  A `cache.json` manifest is updated in the ZIP.

**Future Loads**: When you load this matrix file later, `matrix.eigenvectors()` will detect the presence of these files in the ZIP and load them directly from disk instead of recomputing them.

## Working with Files

### Saving

Since all PyCauset objects are backed by temporary files on creation, "saving" effectively just packages that temporary file into the `.pycauset` ZIP format.

```python
import pycauset

# Create a matrix (backed by a temp file)
M = pycauset.IntegerMatrix(5000, 5000)
M.set(0, 0, 42)

# Save to a permanent location
pycauset.save(M, "my_matrix.pycauset")
```

### Loading

Loading opens the ZIP file and maps the data directly.

```python
# Load the matrix
M_loaded = pycauset.load("my_matrix.pycauset")

print(M_loaded.get(0, 0)) # 42
```

### Temporary Files

When you create a matrix without loading it, PyCauset creates a temporary file to back it.

*   **Location**: By default, these are stored in a `.pycauset` folder in your current working directory.
*   **Changing Location**: You can change this by setting the `PYCAUSET_STORAGE_DIR` environment variable before importing the library.
*   **Cleanup**: These files are **automatically deleted** when the Python object is garbage collected or the script exits. To keep the data, you *must* use `pycauset.save()`.

## Memory Efficiency

PyCauset is highly optimized for the specific types of matrices used in Causal Set Theory.

### Bit Packing
Causal matrices (adjacency matrices) are boolean (0 or 1). PyCauset stores them as **Bit Matrices**, using 1 bit per element.
*   **NumPy `bool`**: 1 byte (8 bits) per element.
*   **PyCauset `BitMatrix`**: 1 bit per element (8x smaller).

### Triangular Storage
Causal matrices are strictly upper triangular (events can only influence future events). PyCauset only stores the upper triangle.
*   **Space Savings**: ~2x smaller than a dense matrix.

**Combined Impact**:
For a causal set of size $N=100,000$:
*   **NumPy (`int8`)**: $100,000^2$ bytes $\approx$ **10 GB**
*   **PyCauset (`TriangularBitMatrix`)**: $\frac{100,000^2}{2 \times 8}$ bytes $\approx$ **625 MB**

> **Performance Note**: While PyCauset makes it *possible* to run simulations with hundreds of thousands of elements on a laptop (which would be impossible with RAM-based arrays), please note that **disk I/O is slower than RAM**. Operations on these massive datasets will take time. A fast NVMe SSD is highly recommended.

## Lazy Evaluation & Metadata Operations

PyCauset uses "lazy evaluation" to perform certain operations instantly, regardless of matrix size. Instead of modifying the massive binary data on disk, we simply update the lightweight `metadata.json`.

### 1. Scalar Multiplication
If you multiply a matrix by a scalar, PyCauset updates the `scalar` field in the metadata.
*   **Operation**: `M_new = M * 2.5`
*   **Result**: The binary data is copied (or referenced), and the `scalar` field becomes `old_scalar * 2.5`.
*   **Access**: When you read an element `M.get(i, j)`, the library reads the raw value and multiplies it by the scalar on the fly.

### 2. Transposition
Transposing a matrix is an $O(1)$ operation.
*   **Operation**: `M_T = M.transpose()`
*   **Result**: The `is_transposed` flag in the metadata is toggled.
*   **Access**: When you read `M.get(i, j)`, the library internally swaps the indices and reads `(j, i)` from the raw data.

This allows you to manipulate the mathematical properties of massive matrices without paying the cost of rewriting gigabytes of data.

## Best Practices

1.  **Use SSDs**: Since memory mapping relies on disk I/O, a fast NVMe SSD will significantly improve performance compared to a mechanical HDD.
2.  **Close Objects**: While Python's garbage collector handles cleanup, explicitly calling `matrix.close()` on a _very_ large file ensures the underlying file handles are released immediately.
3.  **Transposition**: Transposing a matrix (`M.transpose()`) is a metadata-only operation ($O(1)$). It just sets a flag. The data is not moved.
4.  **Context Managers**: Use `with` blocks to ensure files are closed.
    ```python
    with pycauset.load("data.pycauset") as M:
        print(M.size())
    ```
