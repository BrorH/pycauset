# Bug Log

This file documents discovered bugs per the `Testing and Bug Tracking Protocol`
(`documentation/project/protocols/Testing and Bug Tracking.md`).

---

## [Date: 2026-08-30] matrix(storage="disk") silently ignored, benchmarks tested RAM instead of disk

**Status**: Fixed
**Severity**: Medium
**Component**: Python matrix factory (`python/pycauset/_internal/matrix_api.py`)

**Description**:
`pycauset.matrix(data, storage="disk")` silently dropped the `storage` kwarg for
NumPy-array input, always returning an anonymous `:memory:` matrix. The disk I/O
tests (`test_io_consistency.py`) and the out-of-core benchmarks
(`benchmark_outofcore_matmul.py`, `benchmark_numpy_parity.py`) therefore
benchmarked RAM while claiming to exercise disk-backed storage, a silent wrong
answer for the out-of-core (R2_STREAM) claims.

**Reproduction**:
```python
import numpy as np, pycauset as pc
mat = pc.matrix(np.random.rand(100, 100), storage="disk")
mat.get_backing_file()  # was ":memory:" (RAM), expected a .tmp file
```

**Root Cause**:
`Matrix.__new__` only handled `max_in_ram_bytes`; any other kwarg, including
`storage`, was dropped before the `native.asarray` fast path, so the memory
threshold was never lowered and the matrix never spilled.

**Fix**:
`Matrix.__new__` now pops and validates `storage` (`"ram"` or `"disk"`), and for
`"disk"` wraps the `native.asarray` fast path in a zeroed memory threshold so the
matrix spills to a mmap'd file. `storage="disk"` with non-NumPy input raises a
clear `TypeError`, and any other value raises `ValueError`. Regression tests in
`tests/python/test_storage_kwarg.py`; `test_io_consistency.py` now asserts the
backing file is not `:memory:` for the disk cases.

---

## [Date: 2026-08-30] macOS flaky spill of tiny matrices (free-RAM detection reads low)

**Status**: Fixed
**Severity**: High
**Component**: Compute / MemoryGovernor (`src/core/MemoryGovernor.cpp`)

**Description**:
After `matrix(storage="disk")` was made to actually allocate disk-backed matrices,
the macOS CI job became flaky: `test_viz_r2.py::test_top_level_verbs` intermittently
failed with "Export to NumPy is blocked for file-backed/out-of-core objects" because
`causet.C` (a fresh 100 by 100 bit matrix) was wrongly spilled to disk.

**Root Cause**:
`MemoryGovernor::refresh_system_stats()` computed macOS available RAM from
`vm_stat.free_count` (strictly free pages). Free pages exclude the reclaimable file
cache and read artificially low after a build or a large mmap write, so
`request_ram()` returned false for tiny objects and routed them to disk. The Linux
branch already had the equivalent fix (MemAvailable instead of MemFree).

**Fix**:
On macOS, available RAM is now `free_count + inactive_count` (inactive pages are
clean and reclaimable), mirroring the Linux MemAvailable fix. Also closed the
disk-backed matrices in `test_io_consistency.py` so they no longer leak mmap
mappings and file descriptors.

---

## [Date: 2026-08-30] macOS CI hang in matmul (OpenBLAS GEMM thread oversubscription)

**Status**: Fixed
**Severity**: High
**Component**: Compute / CPU matmul (`CpuSolver.cpp`), import-time OpenBLAS setup (`python/pycauset/__init__.py`)

**Description**:
The macOS CI job hung deterministically at
`test_io_consistency.py::test_direct_path_consistency`, a 500 by 500 float64
`matmul` that runs right after a 98 MB test. The suite stopped at 47% and timed
out after 10 minutes. Windows and Linux were unaffected.

**Reproduction**:
The hang was isolated with `tools/macos_hang_diagnose.py`, which prints a line
between each step. On macOS the last line before the timeout was
`STEP direct: matmul(a, b)`, so the stall was inside the OpenBLAS GEMM call, not
matrix construction or `to_numpy`.

**Root Cause**:
`CpuSolver::attempt_direct_path` bumped the OpenBLAS thread count to a hardcoded
20 threads before every double-precision GEMM, then restored it. On the macOS
runner, which exposes only 3 vCPUs, asking OpenBLAS to grow its pthread pool to
20 threads at runtime deadlocked inside `cblas_dgemm`. A secondary bug meant the
import-time thread default (`openblas_set_num_threads(8)`) was never applied on
macOS/Linux because the code probed only the Windows DLL name `libopenblas.dll`.

**Fix**:
Cap the GEMM thread bump to `std::thread::hardware_concurrency()` (at most 20)
and skip the `openblas_set_num_threads` call entirely when the target already
matches the current count, so no pool reconfiguration happens on machines whose
global default already matches the hardware. Also probe the correct OpenBLAS
library name per platform in `__init__.py` and cap the default to
`os.cpu_count()`. The existing
`test_io_consistency.py::test_direct_path_consistency` is the regression test;
the macOS suite now completes 802 passed.

---

## [Date: 2026-08-29] Wrong elementwise results on zero-offset submatrix views (SIMD fast paths)

**Status**: Fixed
**Severity**: Critical
**Component**: Compute / CPU SIMD elementwise (`CpuSolver.cpp`)

**Description**:
The SIMD fast paths (`try_fast_simd`, `binary_op_impl` dense path, `scalar_op_impl`)
decided a `DenseMatrix` was safe to process as a raw flat pointer using only
`has_view_offset()` (offset == 0). A zero-offset submatrix view, e.g. `A[:3,:3]`
sliced from a 5×5 parent, has offset 0 but is *strided*: its storage row length
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
- the "stack buffer overrun" reported at runtime.

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
**Component**: Core / storage (`MemoryMapper` handle lifetime), *not* the lazy routing

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
`MemoryMapper`'s `hFile_`/`hMapping_` members are **uninitialized**, the constructor's
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
the previously-crashing 7-module batch 8× consecutively, all pass.


