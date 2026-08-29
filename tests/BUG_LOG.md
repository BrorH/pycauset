# Bug Log

This file documents discovered bugs per the `Testing and Bug Tracking Protocol`
(`documentation/project/protocols/Testing and Bug Tracking.md`).

---

## [Date: 2026-08-29] Wrong elementwise results on zero-offset submatrix views (SIMD fast paths)

**Status**: Fixed
**Severity**: Critical
**Component**: Compute / CPU SIMD elementwise (`CpuSolver.cpp`)

**Description**:
The SIMD fast paths (`try_fast_simd`, `binary_op_impl` dense path, `scalar_op_impl`)
decided a `DenseMatrix` was safe to process as a raw flat pointer using only
`has_view_offset()` (offset == 0). A zero-offset submatrix view — e.g. `A[:3,:3]`
sliced from a 5×5 parent — has offset 0 but is *strided*: its storage row length
is `base_cols() == 5`, not the logical 3. The fast path therefore read/wrote
`data()[0..9]` contiguously and produced **wrong results** instead of the view's
true (strided) elements.

The eager `__mul__` binding (`av * bv`) passed view operands directly into
`CpuSolver::elementwise_multiply`, hitting this path; `+`/`-` were unaffected in
the common path because they return a lazy expression that is materialized
through the element-wise accessors.

**Reproduction**:
```python
import numpy as np, pycauset as pc
A = np.random.default_rng(7).standard_normal((5, 5))
B = np.random.default_rng(7).standard_normal((5, 5))
a, b = pc.matrix(A), pc.matrix(B)
av, bv = a[0:3, 0:3], b[0:3, 0:3]
np.allclose(np.asarray(av * bv), A[0:3, 0:3] * B[0:3, 0:3])  # False (was)
```

**Root Cause**:
The contiguity guard checked only `!is_transposed() && !has_view_offset()`.
`has_view_offset()` is false for a view whose `row_offset`/`col_offset` are both
zero, even though its logical shape is smaller than its storage shape. The raw
pointer kernels assume `data()[i] == element(i / cols, i % cols)` with `cols ==
base_cols()`, which is false for a zero-offset view.

**Fix**:
Require a **full span** (`rows() == base_rows() && cols() == base_cols()`) in
addition to non-transposed + zero-offset in all three fast paths, so views fall
back to the element-wise (`get`/`set`) path. Regression test added in
`tests/python/test_operations_extensive.py::test_zero_offset_view_elementwise`.

---

## [Date: 2026-08-29] Access violation in mixed-type elementwise ops (null deref in `binary_op_impl` fast path)

**Status**: Fixed
**Severity**: Critical
**Component**: Compute / CPU elementwise (`CpuSolver.cpp`)

**Description**:
While hardening the fast paths for the view bug above, the full-span checks in
`binary_op_impl` were hoisted *before* the null guard, dereferencing `a_dense`/
`b_dense` which are null for mixed-type operands (e.g. `FloatMatrix + IntegerMatrix`
in a `double` dispatch). This crashed with exit code `0xC0000005` (access violation)
— the "stack buffer overrun" reported at runtime.

**Reproduction**:
```python
import pycauset as pc
f, i = pc.FloatMatrix(5), pc.IntegerMatrix(5)
f[0, 1], i[0, 1] = 2.5, 3
_ = f + i  # crashed
```

**Root Cause**:
`a_dense = dynamic_cast<const DenseMatrix<T>*>(&a)` is null when the operand is a
different template type than `T`. Computing `a_dense->rows()` outside the
`a_dense && b_dense` short-circuit guard dereferenced the null pointer.

**Fix**:
Inlined the full-span checks into the `if` condition *after* `a_dense && b_dense &&
a_full && b_full`, so short-circuit evaluation guarantees non-null before the
`base_rows()`/`base_cols()` dereferences. Covered by the existing
`test_operations_extensive.py::test_mixed_type_arithmetic`.

---

## [Date: 2026-08-29] Stack-buffer overrun from lazy elementwise routing (R2_CPU)

**Status**: Fixed (root-caused + re-enabled)
**Severity**: Critical
**Component**: Core / storage (`MemoryMapper` handle lifetime) — *not* the lazy routing

**Description**:
Routing the lazy `A+B` / `A−B` / `A÷B` materialization (`MatrixExpressionWrapper::eval_into`)
straight to the device's SIMD elementwise kernel raised lazy `add` from 0.08× to
0.92× NumPy, but exposed a **stack-buffer overrun / `INVALID_HANDLE`** (`STATUS_STACK_BUFFER_OVERRUN`
0xC0000409 / `STATUS_INVALID_HANDLE` 0xC0000008). It was a Heisenbug: only reproducible
when running several test modules together under `python -m unittest` (after the
memory-spill test and structured-matrix tests set global state), not in-process and
not from a minimal sequence.

**Reproduction**:
```bash
python -m unittest tests.python.test_lazy_evaluation tests.python.test_lazy_ops_comprehensive \
    tests.python.test_elementwise_r2 tests.python.test_operations tests.python.test_operations_extensive \
    tests.python.test_numpy_interop tests.python.test_interop_extensive
# intermittently exits 0xC0000409 / 0xC0000008
```

**Root Cause** (confirmed via procdump + cdb stack trace):
`MemoryMapper`'s `hFile_`/`hMapping_` members are **uninitialized** — the constructor's
initializer list omitted them. For a `:memory:` matrix, `open_file` sets `hFile_ =
INVALID_HANDLE_VALUE` but **never sets `hMapping_`**, so `~MemoryMapper → close_file()`
runs `if (hMapping_) CloseHandle(hMapping_)` on a **garbage pointer**. Depending on the
heap contents at that address the garbage `hMapping_` either read as null (no-op) or as a
non-null invalid handle, producing `STATUS_INVALID_HANDLE` (0xC0000008) or corrupting a
nearby stack cookie (`0xC0000409`). The lazy routing merely shifted the heap layout,
which changed the garbage value and surfaced the latent bug.

Stack (cdb, mini-dump of the first-chance exception):
```
KERNELBASE!CloseHandle+0x49
pycauset_core!MemoryMapper::~MemoryMapper+0x50
pycauset_core!std::_Ref_count_resource<MemoryMapper*>::_Destroy+0x17
pycauset_core!PersistentObject::~PersistentObject+0x6d
_pycauset!pybind11::class_<DenseMatrix<float16_t>>::dealloc+0x6e
```

**Fix**:
Initialize the handles in `MemoryMapper`'s constructor initializer list:
`hFile_(INVALID_HANDLE_VALUE)`, `hMapping_(nullptr)` on Windows, `fd_(-1)` elsewhere.
The lazy-routing change is **re-enabled** (no longer the cause). Verified by running
the previously-crashing 7-module batch 8× consecutively — all pass.


