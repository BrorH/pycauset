# pycauset.Vector

```python
pycauset.Vector(source, dtype=None)
```

The `Vector` class serves as a smart factory for creating vector objects in `pycauset`. It automatically selects the most optimized backend implementation based on the input data or the specified `dtype`.

If `dtype` is specified, it forces the creation of a specific storage dtype (and therefore a specific optimized backend class where available). If `dtype` is NOT specified, `pycauset.Vector(...)` will infer a dtype from the input data.

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

If the dtype cannot be normalized to a known token, `pycauset.Vector` may fall back to a generic Python implementation.

## Return types (optimized backends)

Depending on `dtype`, `pycauset.Vector` returns one of the optimized native vector classes:

- `"int8"` -> [[pycauset.Int8Vector]]
- `"int16"` -> [[pycauset.Int16Vector]]
- `"int32"` -> [[pycauset.IntegerVector]]
- `"int64"` -> [[pycauset.Int64Vector]]
- `"uint8"` -> [[pycauset.UInt8Vector]]
- `"uint16"` -> [[pycauset.UInt16Vector]]
- `"uint32"` -> [[pycauset.UInt32Vector]]
- `"uint64"` -> [[pycauset.UInt64Vector]]
- `"float16"` -> [[pycauset.Float16Vector]]
- `"float32"` -> [[pycauset.Float32Vector]]
- `"float64"` -> [[pycauset.FloatVector]]
- `"complex_float16"` -> [[pycauset.ComplexFloat16Vector]]
- `"complex_float32"` -> [[pycauset.ComplexFloat32Vector]]
- `"complex_float64"` -> [[pycauset.ComplexFloat64Vector]]
- `"bool"` / `"bit"` -> [[pycauset.BitVector]]

All returned vector objects share a common interface. Small vectors (below the configured memory threshold) are stored in RAM for performance, while larger vectors are backed by disk storage.

## Parameters

*   **source** (*int, list, or numpy.ndarray*): 
    *   If an `int` is provided, it creates a new empty (zero-filled) vector of size $N$.
    *   If a list/tuple or NumPy array is provided, it creates a vector populated with that data.
*   **dtype** (*type, optional*): Desired dtype. See “DType tokens” above.

## Returns

*   **Vector**: An instance of a vector class (e.g., `FloatVector`, `IntegerVector`, `BitVector`).

## Optimized Backends

The `Vector` factory returns instances of specialized classes optimized for specific data types. While these classes are not typically instantiated directly, they all share the common `Vector` interface.

*   **[[pycauset.IntegerVector]]**: Dense vector storing 32-bit signed integers.
*   **[[pycauset.Int8Vector]]**, **[[pycauset.Int16Vector]]**, **[[pycauset.Int64Vector]]**: Dense signed integer vectors.
*   **[[pycauset.UInt8Vector]]**, **[[pycauset.UInt16Vector]]**, **[[pycauset.UInt32Vector]]**, **[[pycauset.UInt64Vector]]**: Dense unsigned integer vectors.
*   **[[pycauset.FloatVector]]**: Dense vector storing 64-bit floating-point numbers.
*   **[[pycauset.Float32Vector]]**: Dense vector storing 32-bit floating-point numbers.
*   **[[pycauset.Float16Vector]]**: Dense vector storing 16-bit floating-point numbers.
*   **[[pycauset.ComplexFloat16Vector]]**, **[[pycauset.ComplexFloat32Vector]]**, **[[pycauset.ComplexFloat64Vector]]**: Complex float vectors.
*   **[[pycauset.BitVector]]**: Dense vector storing boolean values (bit-packed).

## Examples

```python
import pycauset

# Create a vector of size 1000 initialized to zeros
v1 = pycauset.Vector(1000, dtype=float)

# Explicit integer widths
v_i8 = pycauset.Vector(10, dtype="int8")
v_u32 = pycauset.Vector(10, dtype="uint32")

# Create an integer vector from a list
v2 = pycauset.Vector([1, 2, 3, 4, 5])

# Create a bit vector from a boolean list
v3 = pycauset.Vector([True, False, True])

# Create from a NumPy array
import numpy as np
arr = np.array([1.5, 2.5, 3.5])
v4 = pycauset.Vector(arr)  # Creates FloatVector

# Complex float vectors
v_c = pycauset.Vector([1+2j, 3-4j], dtype="complex_float32")
```
