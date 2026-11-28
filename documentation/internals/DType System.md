# DType System (Scalar Kinds, Promotion, Overflow)

This document defines the dtype behavior of PyCauset for both matrices and vectors.

The rules here are **authoritative**: if code behaves differently, the code is wrong and should be changed (or this document must be updated as part of an approved change).

## 1) Scalar model

Every stored element type is described by:

- **kind**: `bit` | `int` | `float`
- **width**:
  - `bit`: width is 1
  - `int`: 8/16/32/64 (signed and unsigned)
  - `float`: 16/32/64
- **flags**:
  - `complex`: supported for float base dtypes only
  - `unsigned`: valid only for `int`

Examples:

- `bit` (aka `bool` in the public Python API)
- `int16`, `int32`, `int64`
- `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`
- `complex_float32` (a.k.a. `complex64`)
- `complex_float64` (a.k.a. `complex128`)

### Supported scalar set (current)

- `bit` / `bool`
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`
- `complex_float16`, `complex_float32`, `complex_float64`

Complex permutations of non-float dtypes (`complex int*`, `complex bit`) are intentionally unsupported.

## 2) Fundamental kinds and the “no promote down” rule

PyCauset treats `bit`, `int`, and `float` as **fundamental kinds**.

Rules:

1) PyCauset never **promotes down** across fundamental kinds.
2) If a float participates, the result kind is float.
3) Underpromotion applies **within** a kind, not across kinds.

4) When an integer (or bit) meets a float, the result float must be able to represent
   the integer operand's dtype: bit/int8/uint8 → at least float16, int16/uint16 → at
   least float32, 32/64-bit integers → float64. Otherwise values silently overflow or
   lose precision (e.g. int32 100000 + float16 would be inf).
5) Integer ÷ integer gives at least float32, and float64 when an operand is a 32/64-bit
   integer (`int16 / uint16` is float32, which holds both exactly; `int8 / uint64` is float64,
   as in NumPy: a quotient is a float anyway, so the signed / uint64 error of §7 does not
   apply).
6) `bool + bool`, `bool - bool` and `bool @ bool` count in int32 (bit counting); `bool * bool`
   stays bool (a product of 0/1 values), as in NumPy.

Examples:

- `matmul(bit, float64) -> float64`
- `matmul(float32, float64) -> float32` (default underpromotion within float)
- `add(int32, float16) -> float64`, `add(int8, float16) -> float16`
- `divide(int32, int32) -> float64`, `divide(int16, int16) -> float32`

This prevents nonsensical outcomes like “compute a float matmul but store the result as bit”.

## 3) Underpromotion (within floats)

When PyCauset underpromotes, it means:

- **compute** happens in the selected smallest dtype, and
- **storage** uses that same dtype.

There is no silent widening of intermediates “for accuracy” in the default path.

Linear algebra is the exception: `solve`, `inv`, `det`, `lstsq`, `svd` and the other LAPACK
operations compute in NumPy's LAPACK dtype (`lapack_dtype` in `math/LinearAlgebra.cpp`):
integers and bit in float64, float16 / float32 in float32, a float32 with a float64 operand in
float64 (`solve(float32, float64)` is float64, as in NumPy).

## 3.1) Underpromotion within integers

PyCauset applies the same ethos to integer widths as to floats: if an operation can be executed and stored in a smaller integer width, PyCauset will not silently widen “for safety”.

This is independent from overflow policy (see below): widening is a dtype/semantics decision; it is not an overflow workaround.

### 3.2) Default integer dtype: int32 (a deliberate difference from NumPy)

Data without a dtype of its own gets **int32** where NumPy picks int64: Python integer
literals (`pycauset.vector([1, 2])`, `pycauset.matrix([[1, 2]])`) and the type tokens `int` /
`"int"` (`"uint"` is uint32). Half the memory of int64, values up to 2^31 - 1.

Data that already has a dtype keeps it: a NumPy int64 array imports as int64. PyCauset never
narrows typed input, and it never picks a dtype from the values (1..500 fitting in int16 says
nothing about `x * x`, and element-wise integer results wrap like NumPy's). For smaller
storage pass `dtype="int16"` (or `int8`) explicitly. Overflow is guarded where it is most
likely: integer matmul raises `OverflowError` (§5.1). Choosing widths automatically from
known value ranges is the value-bounds feature (review/features/VALUE_BOUNDS.md).

## 4) Complex numbers

### 4.1 Complex floats

For float32/float64, complex dtypes are first-class numeric types:

- `complex64` (complex float32)
- `complex128` (complex float64)

Where applicable, BLAS-backed complex kernels are used.

`complex_float16` is supported as a first-class dtype and is represented internally as a two-plane float16 storage model (real + imaginary).

### 4.3 Promotion with complex operands

The real rules of §2 and §3 apply to the precision, and a complex operand makes the result
complex:

- float / complex with float / complex: the narrower precision (`complex64 + float64 ->
  complex64`, `complex128 + float32 -> complex64`, `complex_float16 + anything float ->
  complex_float16`); NumPy widens (`complex128`). Warned (`PyCausetDTypeWarning`).
- an integer or bool operand raises the floor to the float that holds it (§2.4):
  `complex_float16 + int16 -> complex64`, `complex64 + int32 -> complex128`,
  `complex128 + int8 -> complex128`.

`tests/engine/test_dtype_rules.py` checks these for every pair.

### 4.2 Complex non-floats (including complex-bit)

Complex permutations of non-float dtypes are a non-goal by design.

- `complex int*` and `complex bit` are not supported.
- Complex support is limited to complex floats.

## 5) Overflow

### 5.1 Integer overflow

Overflow behavior depends on the operation class:

- **Elementwise** (`add`, `sub`, `mul`, `div`, and their scalar variants): integer overflow **wraps silently** (C/NumPy two's-complement semantics). This is the documented, supported behavior: it is not an error and does not auto-promote.
- **Reductions** (`matmul`): PyCauset uses a **wider internal accumulator** and **throws `OverflowError`** when the final result does not fit the output dtype. This keeps reductions free of silent wrap. `dot` is the 1-D case of matmul and follows the same rule (review 3.6 C-9).

For large int32 matmuls PyCauset may also emit an **advisory** `PyCausetOverflowRiskWarning`, estimated from a fixed sample of entries, before computing. It never affects the result.

Reduction-aware accumulator widening is a dtype-policy event (see `PyCausetDTypeWarning`).

### 5.2 Float overflow

Float overflow is possible (e.g., float32 can overflow to `inf`). PyCauset follows IEEE-754 semantics (`inf`/`nan`) by default.

Optional “strict” validation (e.g., finiteness checks) can exist, but it is not the default because scanning 100GB+ outputs is expensive.

## 6) Bit is special (scale-first rules)

Bit matrices/vectors are used to store very large binary structures where widening is often infeasible.

A key fact:

- `bit` is **1 bit / element**
- `int32` is **32 bits / element** (32× larger)
- `float64` is **64 bits / element** (64× larger)
- complex variants are typically **2×** the storage of their base dtype (two planes)

So widening a 100GB bit dataset can easily become multi-terabyte storage.

Because of that, every operation must explicitly declare its `bit` behavior as one of:

- **bitwise**: preserves `bit` and stays bit-packed
- **numeric**: produces non-binary results and therefore widens to `int`/`float`
- **error-by-design**: for `bit` inputs, the operation throws unless the caller explicitly requests a widened dtype/output

This declaration is part of the operation’s contract (and is tested).

## 7) Signed/unsigned mixing

Unsigned integers are supported (`uint8/16/32/64`). Mixing signed and unsigned operands follows
NumPy: the result is the narrowest signed integer that holds both ranges (`uint32 + int32 ->
int64`, `int8 & uint8 -> int16`). No integer holds a signed range and uint64's: NumPy then
gives `float64`, which silently rounds above 2^53, so PyCauset raises instead (decision F-035b;
to be resolved exactly with value bounds, review Phase F1). Bitwise operations raise
`TypeError` there, as NumPy does.

## 9) Matrix with scalar

A scalar takes part in an operation as an operand of one dtype, and the matrix rules above
decide the result (`scalar_result_dtype` in `math/ScalarOps.cpp` calls `promotion::resolve`):

- A **NumPy scalar** (`np.float32(1)`, `np.int64(3)`) is an operand of its own dtype (NumPy's
  "strong" scalars). Within-float under-promotion applies to it too: a scalar never widens a
  float matrix (`float16 M + np.float32(1)` is float16, `float64 M * np.float32(2)` float32);
  NumPy would widen. Whenever a result is narrower than NumPy's, one `PyCausetDTypeWarning`
  says so. This holds for every operand kind (matrices, vectors, NumPy scalars, matrix-vector
  products) in either order.
- A **bool** (`True`, `np.True_`) is a bool operand: `B * True` is bool (as `B * B`), `B + True`
  counts in int32 (as `B + B`), `int8 M * True` is int8, as in NumPy.
- A **Python scalar** is weak: an `int` takes the matrix's dtype (int32 with a bool matrix,
  the default integer; one beyond 64 bits is a float, so `float64 M + 2**64` computes and an
  integer matrix raises `OverflowError`, as NumPy 2), a `float` the matrix's float dtype (float64 with an integer matrix), a
  `complex` the complex of the matrix's precision (complex128 with an integer matrix).
- Hence `int8 M / 1` is float32 (§2.5; NumPy: float64), and an integer matrix with a NumPy
  scalar whose type no integer holds together with it (`int8 M + np.uint64(1)`) raises (§7).
- A Python int that does not fit the matrix's integer dtype raises `OverflowError`, as in
  NumPy.

`tests/engine/test_scalar_ops.py` checks every dtype against these rules, and the values
against NumPy.

## 8) Coverage and enforcement

Which (structure, dtype) pairs exist is decided by one catalogue
(`include/pycauset/core/DTypeDispatch.hpp`, see the Adding Types protocol). What every
`tests/engine/test_dtype_rules.py` checks the result dtype of every pair of real dtypes for
`+ - * / @` against these rules (NumPy, with the differences above written out), independently
of the promotion resolver. What every
operation does on every pair, view and storage mode is enumerated against NumPy by the
operation atlas (`tests/atlas/`, report in `review/atlas/ATLAS.md`); its CI test fails on any
wrong result or regression. (The hand-maintained support matrix it replaced is removed.)
