# pycauset.Matrix

```python
pycauset.Matrix(source, dtype=None)
```

The `Matrix` class serves as a smart factory for creating matrix objects in `pycauset`. It automatically selects the most optimized backend implementation based on the input data or the specified `dtype`.

If `dtype` is specified, it forces the creation of a specific storage dtype (and therefore a specific optimized backend class where available). If `dtype` is NOT specified, `pycauset.Matrix(...)` will infer a dtype from the input data.

For the authoritative dtype rules (promotion, underpromotion, overflow), see `documentation/internals/DType System.md`.

## DType tokens (recommended)

`dtype` accepts simple NumPy-like tokens (strings). These are also exported as public sentinels on the Python module, e.g. `pycauset.int16`, `pycauset.uint32`, `pycauset.complex_float32`.

Supported dtype strings:

- Bit/boolean: `"bit"`, `"bool"`, `"bool_"`
- Signed integers: `"int8"`, `"int16"`, `"int32"`, `"int64"`
- Unsigned integers: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"`
- Floats: `"float16"`, `"float32"`, `"float64"`
- Complex floats: `"complex_float16"`, `"complex_float32"`, `"complex_float64"`

Aliases:

- `"int"` -> `"int32"`
- `"uint"` -> `"uint32"`
- `"float"` -> `"float64"`
- `"complex"` -> `"complex_float64"`
- `"complex64"` -> `"complex_float32"`
- `"complex128"` -> `"complex_float64"`

Shorthand aliases are accepted (case-insensitive): `"i8"`, `"u32"`, `"f16"`, `"f32"`, `"f64"`, `"half"`, `"single"`, `"double"`.

## Other accepted dtype forms

`dtype` can also be:

- Python builtins: `int`, `float`, `bool`
- NumPy dtypes and dtype-like objects: `np.int16`, `np.dtype("uint32")`, `np.float32`, `np.bool_`, `np.complex64`, ...

If the dtype cannot be normalized to a known token, `pycauset.Matrix` may fall back to a generic Python implementation.

## Default precision for large sizes

If you construct an empty matrix with `source` as an integer size (N), PyCauset will default to float storage and may automatically enforce `float32` for very large matrices to reduce storage and I/O.

You can override this behavior via `force_precision="double"|"float64"` or `force_precision="single"|"float32"`.

## Return types (optimized backends)

Depending on `dtype` and structure, `pycauset.Matrix` returns one of the optimized native matrix classes.

Integer/uint dtypes (dense):

- `"int8"` -> [[pycauset.Int8Matrix]]
- `"int16"` -> [[pycauset.Int16Matrix]]
- `"int32"` -> [[pycauset.IntegerMatrix]]
- `"int64"` -> [[pycauset.Int64Matrix]]
- `"uint8"` -> [[pycauset.UInt8Matrix]]
- `"uint16"` -> [[pycauset.UInt16Matrix]]
- `"uint32"` -> [[pycauset.UInt32Matrix]]
- `"uint64"` -> [[pycauset.UInt64Matrix]]

Float/complex dtypes (dense):

- `"float16"` -> [[pycauset.Float16Matrix]]
- `"float32"` -> [[pycauset.Float32Matrix]]
- `"float64"` -> [[pycauset.FloatMatrix]]
- `"complex_float16"` -> [[pycauset.ComplexFloat16Matrix]]
- `"complex_float32"` -> [[pycauset.ComplexFloat32Matrix]]
- `"complex_float64"` -> [[pycauset.ComplexFloat64Matrix]]

Bit/boolean storage:

- `"bool"` / `"bit"` -> [[pycauset.DenseBitMatrix]]

Structure-specialized classes like [[pycauset.TriangularBitMatrix]] and [[pycauset.TriangularIntegerMatrix]] may be returned when creating from data that is detected as strictly upper triangular.

All returned matrix objects share a common interface and are backed by disk storage (except the generic Python fallback).

## Parameters

*   **source** (*int, list, or numpy.ndarray*): 
    *   If an `int` is provided, it creates a new empty (zero-filled) matrix of size $N \times N$.
    *   If a nested list/tuple or NumPy array is provided, it creates a matrix populated with that data.
*   **dtype** (*type, optional*): Desired dtype. See “DType tokens” above.

## Returns

*   **Matrix**: An instance of a matrix class (e.g., `FloatMatrix`, `IntegerMatrix`, `DenseBitMatrix`, or the generic Python `Matrix` fallback).

## Optimized Backends

The `Matrix` factory returns instances of specialized classes optimized for specific data types and structures. These classes are exposed on the Python module (e.g. `pycauset.Int8Matrix`) but are typically created via `pycauset.Matrix(...)`.

*   **[[pycauset.IntegerMatrix]]**: Dense matrix storing 32-bit signed integers.
*   **[[pycauset.Int8Matrix]]**, **[[pycauset.Int16Matrix]]**, **[[pycauset.Int64Matrix]]**: Dense signed integer matrices.
*   **[[pycauset.UInt8Matrix]]**, **[[pycauset.UInt16Matrix]]**, **[[pycauset.UInt32Matrix]]**, **[[pycauset.UInt64Matrix]]**: Dense unsigned integer matrices.
*   **[[pycauset.FloatMatrix]]**: Dense matrix storing 64-bit floating-point numbers.
*   **[[pycauset.Float32Matrix]]**: Dense matrix storing 32-bit floating-point numbers.
*   **[[pycauset.Float16Matrix]]**: Dense matrix storing 16-bit floating-point numbers.
*   **[[pycauset.ComplexFloat16Matrix]]**, **[[pycauset.ComplexFloat32Matrix]]**, **[[pycauset.ComplexFloat64Matrix]]**: Complex float matrices.
*   **[[pycauset.DenseBitMatrix]]**: Dense matrix storing boolean values (bit-packed).
*   **[[pycauset.TriangularIntegerMatrix]]**: Strictly upper triangular integer matrix.
*   **[[pycauset.TriangularFloatMatrix]]**: Strictly upper triangular float matrix.
*   **[[pycauset.TriangularBitMatrix]]**: Strictly upper triangular boolean matrix (bit-packed).

## Examples

```python
import pycauset

# Create a 5x5 empty integer matrix (defaults to int32)
M = pycauset.Matrix(5, dtype=int)

# Explicitly sized integer dtypes
M_i8 = pycauset.Matrix(5, dtype="int8")
M_u64 = pycauset.Matrix(5, dtype="uint64")

# Create a dense boolean matrix (DenseBitMatrix)
M_bool = pycauset.Matrix(5, dtype=bool)

# Create an integer matrix from data
data = ((1, 2), (3, 4))
M_int = pycauset.Matrix(data)
# Returns an optimized IntegerMatrix

# Create a complex matrix
M_c = pycauset.Matrix(((1+2j, 0), (0, 3-4j)), dtype="complex_float32")

# Create a triangular matrix
tri_data = ((0, 1), (0, 0))
M_tri = pycauset.Matrix(tri_data)
# Returns an optimized TriangularIntegerMatrix
```
