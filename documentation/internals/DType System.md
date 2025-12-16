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

Examples:

- `matmul(bit, float64) -> float64`
- `matmul(float32, float64) -> float32` (default underpromotion within float)

This prevents nonsensical outcomes like “compute a float matmul but store the result as bit”.

## 3) Underpromotion (within floats)

When PyCauset underpromotes, it means:

- **compute** happens in the selected smallest dtype, and
- **storage** uses that same dtype.

There is no silent widening of intermediates “for accuracy” in the default path.

## 3.1) Underpromotion within integers

PyCauset applies the same ethos to integer widths as to floats: if an operation can be executed and stored in a smaller integer width, PyCauset will not silently widen “for safety”.

This is independent from overflow policy (see below): widening is a dtype/semantics decision; it is not an overflow workaround.

## 4) Complex numbers

### 4.1 Complex floats

For float32/float64, complex dtypes are first-class numeric types:

- `complex64` (complex float32)
- `complex128` (complex float64)

Where applicable, BLAS-backed complex kernels are used.

`complex_float16` is supported as a first-class dtype and is represented internally as a two-plane float16 storage model (real + imaginary).

### 4.2 Complex non-floats (including complex-bit)

Complex permutations of non-float dtypes are a non-goal by design.

- `complex int*` and `complex bit` are not supported.
- Complex support is limited to complex floats.

## 5) Overflow

### 5.1 Integer overflow

Integer overflow is a **hard error**.

- PyCauset does not silently wrap.
- PyCauset does not silently widen storage to avoid overflow.

For high-risk ops such as large integer matmul, PyCauset may emit an **advisory risk warning** based on conservative bounds (see `PyCausetOverflowRiskWarning`).

For integer reductions (e.g., `dot`/`matmul`), PyCauset may use a wider internal accumulator dtype to keep overflow behavior defined. This is a dtype-policy event (see `PyCausetDTypeWarning`), and overflow still throws when storing the final output.

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

Unsigned integers are supported (`uint8/16/32/64`).

When mixing signed and unsigned integer operands, PyCauset selects a supported result dtype that can represent the required numeric range.

Important consequences:

- Result dtypes are restricted to the supported widths (8/16/32/64). PyCauset will not invent intermediate 33-bit or 65-bit integers.
- Some mixed signed/unsigned operations will promote to the next wider signed integer type.

Example (typical outcome):

- `uint32 + int32 -> int64`

## 8) Support matrix + enforcement

PyCauset's dtype coverage is declared and enforced via an executable support matrix (ops x dtypes x rank).

- The declared matrix lives in `pycauset._internal.support_matrix`.
- The enforcement test is `tests/python/test_support_matrix.py`.

User-facing documentation should describe the intended behavior, but the support matrix is the no-regressions gate that ensures what is claimed is actually implemented.
