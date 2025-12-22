# R1 DTypes (Integers, Float16, Complex)

Release 1 makes the dtype system explicit and enforceable: integer widths (signed + unsigned), float16 as a first-class dtype, and complex floats end-to-end on CPU.

This guide summarizes the user-facing behavior: what dtypes exist, how promotion behaves, and what happens on overflow.

## Supported dtypes (Release 1)

Across the core matrix/vector surface, the supported scalar set is:

- `bool` / `bit` (bit-packed storage for dense boolean matrices)
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`
- `complex_float16`, `complex_float32`, `complex_float64`

Notes:

- Complex support is **complex floats only**. `complex int*` and `complex bit` are intentionally unsupported.
- `complex_float16` is supported as a first-class dtype (internally implemented via a two-plane float16 storage model).

See also: [[internals/DType System.md|DType System]] (authoritative rules).

## Dtype selection and “underpromotion”

PyCauset follows a “smallest type” ethos.

Within floats, mixed precision underpromotes by default:

- `float32` combined with `float64` uses `float32` compute + storage unless you explicitly request otherwise.

This is a semantics choice (not an overflow workaround) and may emit a dtype-policy warning.

## Overflow behavior (integers)

Integer overflow is a **hard error**.

- PyCauset does not silently wrap.
- PyCauset does not silently widen output storage to avoid overflow.

### Overflow-risk warning for large integer matmul

For high-risk integer reductions like large `matmul`, PyCauset may emit a conservative warning when overflow looks plausible:

- category: `PyCausetOverflowRiskWarning`

This warning is advisory (it can over-warn), but it helps you catch “obviously impossible in this dtype” workloads earlier.

### Reduction-aware accumulator widening

For integer reductions (`dot`/`matmul`), PyCauset may use a wider **internal accumulator** dtype to keep overflow behavior defined in the hot loop.

- category: `PyCausetDTypeWarning`
- important: output storage dtype does **not** automatically change; overflow still throws on the final cast to the requested output dtype.

## Bit matrices are special

`bit` is extremely storage-efficient (1 bit/element), so widening a huge bit dataset can be infeasible.

Each operation must explicitly define what it means on `bit`:

- bitwise behavior (stays bit-packed)
- numeric behavior (widens to `int`/`float`)
- error-by-design unless you explicitly request a widened dtype

## Practical example

```python
import numpy as np
import pycauset as pc

# Unsigned integers
A = pc.matrix(np.array(((1, 2), (3, 4)), dtype=np.uint32))

# Complex float
Z = pc.matrix(np.array(((1 + 2j, 0), (0, 3 - 4j)), dtype=np.complex64))

# float16 allocation
H = pc.ones((2, 2), dtype="float16")
```

## See also

- [[guides/Numpy Integration.md|NumPy Integration]]
- [[internals/DType System.md|DType System]]
- [[dev/Warnings & Exceptions.md|Warnings & Exceptions]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
