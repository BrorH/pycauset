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

A `.pycauset` file is a **single-file binary container** designed for mmap-friendly payload access and sparse, typed metadata.

At a high level:

1.  A fixed-size header selects the active header slot (A/B) to locate the payload and metadata.
2.  The payload is raw, aligned binary data (so it can be memory-mapped efficiently).
3.  Metadata is stored as a sparse, typed block that can be updated without shifting the payload.

### Metadata Example

Conceptually, the metadata includes fields like:

```json
{
  "rows": 1000,
  "cols": 1000,
  "seed": 12345,
  "scalar": 1.0,
  "is_transposed": false,
  "is_conjugated": false,
  "properties": {
    "is_unitary": true,
    "is_hermitian": false
  },
  "cached": {
    "trace": {
      "value": 1000.0
    },
    "determinant": {
      "value": 1.0
    }
  },
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
*   **`is_conjugated`**: A boolean flag indicating if the matrix is logically conjugated (a metadata view).
*   **`properties`**: A dictionary of user-facing **gospel** properties (semantic assertions like `is_unitary`). These are not truth-validated. For boolean-like keys, tri-state semantics are represented by key presence: unset means the key is absent; explicit `False` is stored as `false`.
*   **`cached`**: A dictionary of **cached-derived** values stored alongside validity metadata (e.g., `trace`, `determinant`). On load, valid cached-derived entries are surfaced into `obj.properties` under clean names; invalid/stale entries are ignored.

## Release 1 semantics (snapshots, mutation, caches)

This section is the **canonical** Release 1 documentation for snapshot immutability and caching behavior.

### Terms

- **Snapshot**: a persisted `.pycauset` file on disk, treated as an immutable baseline.
- **Working copy**: a runtime object that may diverge from its source snapshot after mutation.
- **Dirty**: a working copy with payload writes that are not yet persisted.
- **Cached-derived value**: a value that can be recomputed from payload + view-state and must be validated before reuse.
- **Big blob cache**: a cached-derived value that must refer to another `PersistentObject` because it is too large to store directly in typed metadata (e.g., an inverse matrix).

### Power users: gospel `properties`

`obj.properties` is a power-user escape hatch that can intentionally change algorithm choices.

- **Gospel assertions are authoritative:** if you set `obj.properties["is_upper_triangular"] = True`, structure-aware algorithms are allowed to behave *as if* entries below the diagonal are zero.
- **No truth validation:** PyCauset does not scan payload data to confirm your assertions.
- **Tri-state booleans via key presence:**
  - unset means the key is **absent** from the mapping,
  - explicit `False` means the key is present with value `False`.

Examples (Release 1):

- `matmul`: if you assert `is_diagonal` or `is_upper_triangular`/`is_lower_triangular`, multiplication is allowed to treat out-of-structure entries as zero and route structured fast paths.
- `solve`: `solve(a, b)` will use `solve_triangular(a, b)` when `a` is marked diagonal/triangular.
- `eigvalsh`: consults/seeds cached `eigenvalues`; if `is_hermitian` is explicitly `False`, it rejects.

Note: `obj.properties` also surfaces some **cached-derived** values (like `trace`, `determinant`, `norm`, `sum`) for convenience, but these are treated differently:

- cached-derived values are used only when their validity signature matches the current payload + view-state,
- otherwise they are ignored and recomputed on demand.

## Compute-Once Caching

PyCauset implements a "compute-once" philosophy for expensive mathematical operations.

Cache validity is strict: cached values are used only when they are known to match the object’s current payload + view-state. If an object’s payload is mutated, affected caches are discarded.

Note: the on-disk representation splits user-facing metadata into two parts:

- `properties` holds gospel assertions.
- `cached.*` holds cached-derived values plus validity signatures.

Runtime access stays unified via `obj.properties`.

### Cache validity identity (R1, $O(1)$)

Cached-derived values are used only if their signature matches the current object state.

- `payload_uuid`: a persisted snapshot identity for the payload bytes (changes when payload bytes are persisted).
- `view_signature`: a compact signature derived from view-state (`is_transposed`, `is_conjugated`, `scalar`).

These allow deterministic cache reuse decisions without scanning payload.

## Snapshot semantics (mutating loaded files)

PyCauset is designed so you can:

- load a persisted matrix that already has expensive cached artifacts (inverse, eigenvectors, etc.),
- do exploratory one-off edits,
- and not accidentally destroy the on-disk snapshot.

Policy (Release 1):

- A `.pycauset` file is treated as an **immutable snapshot**.
- `pycauset.load(path)` returns a snapshot-backed object.
- **Payload mutation does not implicitly write back** to `path`.
- To persist payload changes, you must explicitly save a new snapshot.

Implementation:

- Implementation uses copy-on-write working copies so small edits do not overwrite the on-disk snapshot.

### Scalar cached values (Trace, Determinant)
When you compute cached-derived scalars like `trace()` or `determinant()`, the result may be stored in the matrix object's memory.
*   **Automatic Persistence**: When you call `pycauset.save(matrix, path)`, cached-derived values may be written into the file’s typed metadata.
*   **Automatic Restoration**: When you `load()` the matrix later, these values are read from metadata, making them available instantly without recomputation.

In Release 1, this same “small cached-derived” mechanism also applies to values like `norm` (when available for the object type).

### Large cached artifacts (Inverse)
For large results that are matrices themselves (like an inverse), PyCauset supports optional persistence to avoid repeating expensive computation.

```python
# Compute an inverse and persist it as a big-blob cache (FloatMatrix only)
inv = matrix.invert(save=True)
```

Big blob cache rule:

- If a cached value needs to refer to another `PersistentObject` (because it is too large for typed metadata), it is treated as a **big blob cache**.
- Big blob caches are persisted as **independent `.pycauset` objects** and the base object stores only a typed reference under `cached.*`.
- In R1 these big-blob objects live next to the base snapshot in `BASE.pycauset.objects/<object_id>.pycauset`.

Failure behavior:

- If a referenced big-blob object is missing or corrupt, PyCauset treats it as a cache miss, emits a storage warning, and recomputes.

Note:

- General eigensolver APIs (`eig`, `eigvals`, `eigvals_arnoldi`, etc.) are not available yet in pre-alpha builds.

## See also
- [[internals/MemoryArchitecture.md|MemoryArchitecture]]
- [[project/protocols/Adding Operations.md|Protocol: Adding Operations]]

## Release 1 container format (on-disk)

This section is the **canonical** Release 1 on-disk container format for `.pycauset` files.

### Goals (R1)

- **Memory-map friendly payload**: large payload bytes live at a stable offset.
- **Sparse, typed metadata**: metadata round-trips missing vs explicit values.
- **Deterministic load**: load selects an active header slot and validates CRCs; no scanning.
- **Crash-safe metadata updates**: A/B header slots select the newest valid metadata pointer.

### Endianness

- R1 containers are **little-endian only**.
- A header endian marker lets readers fail fast.

### Alignment

- `payload_offset` is aligned to **4096 bytes**.
- `metadata_offset` is aligned to **16 bytes**.

### Fixed header (4096 bytes)

The file begins with a fixed **4096-byte header region**:

- A 16-byte preamble.
- Two 128-byte header slots (A and B).
- Remaining bytes are reserved (zero in R1).

#### Preamble (offset 0)

| Field | Type | Notes |
|---|---:|---|
| `magic` | 8 bytes | ASCII `PYCAUSET` |
| `format_version` | u32 | R1 = 1 |
| `endian` | u8 | 1 = little-endian |
| `header_bytes` | u16 | R1 = 4096 |
| `reserved0` | u8[1] | must be 0 |

#### Header slots (A and B)

Each slot is 128 bytes and stores the authoritative pointers:

| Field | Type | Notes |
|---|---:|---|
| `generation` | u64 | monotonic; higher wins |
| `payload_offset` | u64 | aligned to 4096 |
| `payload_length` | u64 | bytes |
| `metadata_offset` | u64 | aligned to 16 |
| `metadata_length` | u64 | bytes |
| `hot_offset` | u64 | 0 in R1 |
| `hot_length` | u64 | 0 in R1 |
| `slot_crc32` | u32 | CRC32 of the first 7 fields (56 bytes) |
| `slot_reserved` | u8[68] | must be 0 |

**Slot validity** (R1):

- `slot_crc32` matches
- offsets/lengths are in-range for file size
- alignment constraints satisfied

**Active slot selection**:

- Choose the valid slot with the highest `generation`.
- If neither slot is valid, loading fails.

### Payload region

The payload is a raw backing store suitable for memory mapping.

- Starts at `payload_offset`.
- Spans `payload_length` bytes.
- Interpretation is defined by identity metadata (rows/cols/matrix_type/data_type/payload_layout).

### Metadata blocks (append-only)

Metadata is stored as one or more blocks after the payload. The active header slot points to the authoritative block.

#### Metadata framing

At `metadata_offset`:

| Field | Type | Notes |
|---|---:|---|
| `block_magic` | 4 bytes | ASCII `PCMB` |
| `block_version` | u32 | R1 = 1 |
| `encoding_version` | u32 | typed-metadata encoding version; R1 = 1 |
| `reserved0` | u32 | must be 0 |
| `payload_length` | u64 | bytes of encoded metadata payload |
| `payload_crc32` | u32 | CRC32 of encoded metadata payload |
| `reserved1` | u32 | must be 0 |
| `payload` | bytes | length = `payload_length` |

If framing or CRC fails, loading fails deterministically.

### Typed metadata encoding v1 (R1)

The encoded metadata payload is a single top-level **Map** with string keys.

Reserved namespaces:

- `view`: system-managed view-state
- `properties`: user-facing gospel assertions (tri-state booleans via key presence)
- `cached`: cached-derived values plus validity metadata
- `provenance`: optional non-semantic provenance

Readers ignore unknown keys.

### Crash-consistent update rule (metadata)

To update metadata without scanning:

1) Append the new metadata block to the end of the file.
2) Ensure it is fully written (and flushed if applicable).
3) Write the inactive header slot with `generation = active.generation + 1` and the new metadata pointer.
4) Optionally flush the header region.

This guarantees $O(1)$ load with no "search for the last valid block".

## Debugging runbook (R1 containers)

When a `.pycauset` file fails to load, the goal is to diagnose it **without scanning payload**.

Checklist:

1) Confirm the file is the R1 container (magic `PYCAUSET`).
2) Inspect header slot A/B:
  - CRC valid?
  - offsets/lengths in-range?
  - alignments satisfied (`payload_offset % 4096 == 0`, `metadata_offset % 16 == 0`)?
  - which slot is active (highest generation among valid slots)?
3) Validate the metadata pointer from the active slot:
  - `metadata_offset + metadata_length` is within file size
  - metadata block magic is `PCMB`
  - metadata block/encoding versions are supported
  - metadata payload CRC32 matches
4) Validate the payload pointer:
  - `payload_offset + payload_length` is within file size
  - payload is mmap-friendly at `payload_offset`

Developer tool:

- `python/pycauset/_internal/storage_debug.py` exposes `summarize_container(path)` which returns a best-effort header summary (preamble + slot A/B + active slot selection).

This tool is exercised by `tests/python/test_storage_debug_tool.py`.

## Working with Files

### Saving

Since all PyCauset objects are backed by files on creation, "saving" typically means copying/linking the backing `.pycauset` container to a permanent location.

```python
import pycauset

# Create a matrix (backed by a temp file)
M = pycauset.zeros((5000, 5000), dtype=pycauset.int32)
M.set(0, 0, 42)

# Save to a permanent location
pycauset.save(M, "my_matrix.pycauset")
```

### Loading

Loading opens the `.pycauset` container and memory-maps the payload directly.

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

PyCauset uses "lazy evaluation" to perform certain operations instantly, regardless of matrix size. Instead of modifying the massive binary data on disk, we simply update lightweight metadata.

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
