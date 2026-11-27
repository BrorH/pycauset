# NumPy Integration Guide

PyCauset works with NumPy: convert a matrix to a NumPy array and back. PyCauset keeps its own storage (RAM or disk), but the two interoperate directly.

## Converting NumPy Arrays to PyCauset

You can convert NumPy arrays into `pycauset` objects using [[docs/functions/pycauset.matrix.md|pycauset.matrix]] and [[docs/functions/pycauset.vector.md|pycauset.vector]]. These constructors automatically detect the data type of the NumPy array and create the corresponding optimized object.

Note: PyCauset does **not** expose a `pycauset.asarray` API. In PyCauset, “arrays” are not a first-class concept; matrices and vectors are.

Rectangular 2D arrays are supported for dense numeric matrices (int/uint/float/complex). Boolean 2D arrays are bit-packed (`DenseMatrix` (dtype `bool`)) and also support rectangular `(rows, cols)` shapes.

Supported dtypes include:

- Integers: `int8/int16/int32/int64` and `uint8/uint16/uint32/uint64`
- Floats: `float16/float32/float64`
- Complex floats: `complex64/complex128` (mapped to `complex_float32/complex_float64`)
- Booleans: `bool_` (mapped to bit-packed storage)

**Performance Note**: Import uses a parallelized direct path for large arrays, achieving >10GB/s on modern hardware. Non-contiguous arrays (slices) are automatically handled via an optimized parallel copy.

```python
import numpy as np
import pycauset as pc

# Convert 1D NumPy array to Vector
arr_1d = np.array([1.0, 2.0, 3.0])
vec = pc.vector(arr_1d)  # Returns [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `float64`)

# Convert 2D NumPy array to Matrix
arr_2d = np.array(((1, 2), (3, 4)), dtype=np.int32)
mat = pc.matrix(arr_2d)  # Returns [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `int32`)

# Unsigned integers
arr_u = np.array(((1, 2), (3, 4)), dtype=np.uint32)
mat_u = pc.matrix(arr_u)  # Returns [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `uint32`)

# Complex
arr_c = np.array(((1 + 2j, 0), (0, 3 - 4j)), dtype=np.complex64)
mat_c = pc.matrix(arr_c)  # Returns [[docs/classes/matrix/pycauset.DenseMatrix.md|pycauset.DenseMatrix]] (dtype `complex_float32`)

# Convert Boolean array
arr_bool = np.array([True, False], dtype=bool)
vec_bool = pc.vector(arr_bool)  # Returns [[docs/classes/vector/pycauset.DenseVector.md|pycauset.DenseVector]] (dtype `bool`)
```

**Note**: This operation creates a **copy** of the data. Depending on the size and the configured memory threshold, the new object will be stored in RAM or on disk. See [[guides/Storage and Memory|guides/Storage and Memory]].

## NumPy UFuncs

NumPy universal functions (`np.sin`, `np.exp`, ...) are **not** supported on PyCauset
matrices: they raise `TypeError` rather than silently pulling a possibly huge matrix into
RAM. Use PyCauset's own operators and functions, or convert explicitly:

```python
import numpy as np
import pycauset as pc

A = pc.matrix(np.random.rand(100, 100))
B = pc.matrix(np.sin(pc.to_numpy(A)))   # explicit, in RAM
```

## Mixed Arithmetic and Ergonomics

PyCauset supports mixed operations between NumPy arrays and PyCauset objects.
Operators (`+`, `-`, `*`, `@`) automatically route to the optimized PyCauset implementation when possible.

```python
A = pc.matrix(np.random.rand(1000, 1000))
B = np.random.rand(1000, 1000)

# Works efficiently!
# PyCauset handles the add, returning a DenseMatrix (float64)
C = A + B 

# Also works (Reverse add support)
D = B + A
```

## Converting PyCauset Objects to NumPy

All `pycauset` Matrix and Vector classes implement the NumPy array protocol (`__array__`). This means you can pass any `pycauset` object directly to `np.array()` or any function that expects an array-like object.

```python
v = pc.vector([1, 2, 3])

# Convert to NumPy array
arr = np.array(v)

# Use in NumPy functions
mean_val = np.mean(v)
std_val = np.std(v)
```

### One copy, always

Conversion is always one copy into a new NumPy array, made by the engine for every class,
dtype and view (a bit matrix becomes a `bool` array, a transposed view its transpose). There
are no NumPy views of PyCauset storage: NumPy could not respect bit packing, triangular
storage or a pending scale. `pc.to_numpy(M, dtype=...)` converts the copy.

### Safety rules (materialization)

Converting a massive out-of-core matrix to NumPy is dangerous-it forces the entire dataset into RAM, which can crash your process. PyCauset guards against this.

- **Snapshot-backed** (`.pycauset`) and **RAM-backed** (`:memory:`) objects: `np.array(obj)` is allowed and returns a copy.
- **Size guard**: `np.array(obj)` raises when the array would not fit in free RAM, or exceeds the ceiling set with [[docs/functions/pycauset.set_export_max_bytes.md|pc.set_export_max_bytes(bytes)]]. RAM- and file-backed objects are treated alike (spilling to a file is automatic, so it is no reason to refuse). Opt out with `pc.to_numpy(obj, allow_huge=True)`.

If you see an export error, either downsize, keep the data in PyCauset ops, or opt in with `allow_huge=True` intentionally.

### On-disk conversions (NumPy formats)

If you need to move data between PyCauset snapshots and NumPy container files, use [[docs/functions/pycauset.convert_file.md|pc.convert_file]].

Important note: exporting from `.pycauset` to `.npy`/`.npz` still produces a dense NumPy array in-process today (guarded by `allow_huge`), because NumPy’s writers expect dense arrays.

- Supported formats: `.pycauset`, `.npy`, `.npz` (import/export in any direction).
- `npz_key` selects a named array inside an archive; defaults to the first key.
- Exports honor the same size guard (`allow_huge=True` to opt out).

Example:

```python
# Snapshot -> npy -> snapshot round-trip
pc.save(pc.matrix(np.eye(3)), "A.pycauset")
pc.convert_file("A.pycauset", "A.npy")
pc.convert_file("A.npy", "A_roundtrip.pycauset")

# Pick a specific array inside an npz
np.savez("bundle.npz", vector0=np.arange(4.0))
pc.convert_file("bundle.npz", "vec.pycauset", npz_key="vector0")
```

## Mixed Arithmetic

You can perform arithmetic operations directly between `pycauset` objects and NumPy arrays. `pycauset` handles the interoperability automatically.

Important note: when you mix a `pycauset` object with a NumPy array, the NumPy side is typically converted to a temporary `pycauset` object and the operation is executed through PyCauset's dtype rules (promotion, underpromotion, overflow). See `documentation/internals/DType System.md`.

### Vector + NumPy Array

```python
v = pc.vector([1, 2, 3])
arr = np.array([10, 20, 30])

# Result is a pycauset Vector (operation happens in C++ backend)
result = v + arr  # [11, 22, 33]
```

### Matrix @ NumPy Vector

You can use NumPy arrays as operands in matrix multiplication.

```python
M = pc.matrix(((1, 0), (0, 1))) # Identity
v_np = np.array([5.0, 6.0])

# Result is a pycauset Vector
v_result = M @ v_np  # [5.0, 6.0]
```

## Performance Considerations

*   **PyCauset as Primary**: When you perform operations like `pycauset_obj + numpy_obj`, `pycauset` attempts to handle the operation. The NumPy array is temporarily converted to a `pycauset` object (backed by RAM or a temporary file), and the operation runs using the optimized C++ backend. The result is a new `pycauset` object.
*   **NumPy as Primary**: If you use a NumPy function like `np.add(pycauset_obj, numpy_obj)`, NumPy will convert the `pycauset` object to an in-memory array first. This might be slower and memory-intensive for large datasets.

**Best Practice**: For massive datasets, stick to `pycauset` native operations and objects as much as possible, only converting to NumPy for small results or specific analysis steps that `pycauset` doesn't yet support.

## See also

- [[docs/functions/pycauset.convert_file.md|pycauset.convert_file]]
- [[docs/functions/pycauset.to_numpy.md|pycauset.to_numpy]]
- [[docs/functions/pycauset.set_export_max_bytes.md|pycauset.set_export_max_bytes]]
- [[docs/functions/pycauset.save.md|pycauset.save]] / [[docs/functions/pycauset.load.md|pycauset.load]]
- [[guides/Storage and Memory|Storage and Memory]]
