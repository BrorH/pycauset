# NumPy Integration Guide

`pycauset` is designed to work seamlessly with the Python scientific stack, particularly NumPy. While `pycauset` uses its own optimized storage (RAM or disk-backed) for handling massive datasets, it provides smooth interoperability with NumPy arrays for convenience and flexibility.

## Converting NumPy Arrays to PyCauset

You can convert NumPy arrays into `pycauset` objects using [[docs/functions/pycauset.matrix.md|pycauset.matrix]] and [[docs/functions/pycauset.vector.md|pycauset.vector]]. These constructors automatically detect the data type of the NumPy array and create the corresponding optimized object.

Note: PyCauset does **not** expose a `pycauset.asarray` API. In PyCauset, “arrays” are not a first-class concept; matrices and vectors are.

Rectangular 2D arrays are supported for dense numeric matrices (int/uint/float/complex). Boolean 2D arrays are bit-packed (`DenseBitMatrix`) and also support rectangular `(rows, cols)` shapes.

Supported dtypes include:

- Integers: `int8/int16/int32/int64` and `uint8/uint16/uint32/uint64`
- Floats: `float16/float32/float64`
- Complex floats: `complex64/complex128` (mapped to `complex_float32/complex_float64`)
- Booleans: `bool_` (mapped to bit-packed storage)

```python
import numpy as np
import pycauset as pc

# Convert 1D NumPy array to Vector
arr_1d = np.array([1.0, 2.0, 3.0])
vec = pc.vector(arr_1d)  # Returns [[pycauset.FloatVector]]

# Convert 2D NumPy array to Matrix
arr_2d = np.array(((1, 2), (3, 4)), dtype=np.int32)
mat = pc.matrix(arr_2d)  # Returns [[pycauset.IntegerMatrix]]

# Unsigned integers
arr_u = np.array(((1, 2), (3, 4)), dtype=np.uint32)
mat_u = pc.matrix(arr_u)  # Returns [[pycauset.UInt32Matrix]]

# Complex
arr_c = np.array(((1 + 2j, 0), (0, 3 - 4j)), dtype=np.complex64)
mat_c = pc.matrix(arr_c)  # Returns [[pycauset.ComplexFloat32Matrix]]

# Convert Boolean array
arr_bool = np.array([True, False], dtype=bool)
vec_bool = pc.vector(arr_bool)  # Returns [[pycauset.BitVector]]
```

**Note**: This operation creates a **copy** of the data. Depending on the size and the configured memory threshold, the new object will be stored in RAM or on disk. See [[guides/Storage and Memory|guides/Storage and Memory]].

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

**Safety rules (materialization)**

- **Snapshot-backed** (`.pycauset`) and **RAM-backed** (`:memory:`) objects: `np.array(obj)` is allowed and returns a copy.
- **Spill/file-backed** objects (e.g., `.tmp`): `np.array(obj)` **raises** by default to prevent surprise full materialization. Opt in explicitly via `pc.to_numpy(obj, allow_huge=True)` if you truly want to load it into RAM.
- **Ceiling control**: [[docs/functions/pycauset.set_export_max_bytes.md|pc.set_export_max_bytes(bytes_or_None)]] sets a materialization limit. `None` disables the size ceiling; file-backed objects still require `allow_huge=True`.

If you see an export error, either downsize, keep the data in PyCauset ops, or opt in with `allow_huge=True` intentionally.

### On-disk conversions (NumPy formats)

If you need to move data between PyCauset snapshots and NumPy container files, use [[docs/functions/pycauset.convert_file.md|pc.convert_file]].

Important note: exporting from `.pycauset` to `.npy`/`.npz` still produces a dense NumPy array in-process today (guarded by `allow_huge`), because NumPy’s writers expect dense arrays.

- Supported formats: `.pycauset`, `.npy`, `.npz` (import/export in any direction).
- `npz_key` selects a named array inside an archive; defaults to the first key.
- Exports honor the same materialization guard: spill/file-backed sources require `allow_huge=True`.

Example:

```python
# Snapshot -> npy -> snapshot round-trip
pc.convert_file("A.pycauset", "A.npy")
pc.convert_file("A.npy", "A_roundtrip.pycauset")

# Pick a specific array inside an npz
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
