# Bug Log

This file documents discovered bugs per the `Testing and Bug Tracking Protocol`
(`documentation/project/protocols/Testing and Bug Tracking.md`).

---

## [Date: 2026-11-11] Physics layer falls back to dense NumPy for every causal-set operation

**Status**: Open (release blocker)
**Severity**: Critical
**Component**: Physics layer (`python/pycauset/causet.py`, `field.py`, `synthetic.py`, `vis.py`) and Python bindings (`TriangularBitMatrix`)

**Description**:
The engine stores causal matrices as bit-packed, disk-spillable `TriangularBitMatrix`
objects, which is the reason PyCauset exists. The physics layer throws that away: every
causal-structure method, the field layer, the synthetic generators and three plotting paths
convert the whole n x n matrix to a dense NumPy array (`np.asarray(self._matrix)` /
`np.array(causet.C)`) and then work in NumPy. This is not a to-NumPy export the user asked
for; it is a silent fallback inside operations that should stay in the engine. Consequences:

- Memory is O(n^2) bytes (bool) or 8-16x that (float64 / complex128), so large causets,
  the headline use case, cannot be analysed at all.
- The conversion itself is a per-element `get(i, j)` loop in `bind_matrix.cpp` (~4M
  elements/s), so it is slow even when memory suffices.
- Some operations repeat the conversion per call (e.g. `past(x)` converts n^2 elements to
  return one column).

Measured on Linux (4 cores), `MinkowskiDiamond(2)`:

| n | sprinkle (native) | `relation_fraction()` | `past(x)` |
| --: | --: | --: | --: |
| 2,000 | 0.02 s | 1.1 s | 1.1 s |
| 10,000 | 0.55 s | 27 s | 28 s |
| 20,000 | 2.3 s | 105 s (835 MB peak) | 107 s |

Extrapolated: ~45 min and ~10 GB per call at n = 100,000 (the `demos/06_large_n.py` size).

**Affected operations** (each is a case in `tests/python/test_no_dense_fallback.py`):

- `CausalSet`: `relation_fraction`, `myrheim_meyer_dimension`, `past`, `future`,
  `interval`, `is_chain`, `is_antichain`, `longest_chain`, `layers`, `links`, `validate`
  (`validate_causal_matrix` also does an O(n^3) dense `uint8` matmul; `links` likewise).
- Plotting: `plot_hasse` and `plot_causal_matrix` densify the *whole* matrix before taking
  their `max_points` subset; `plot_embedding(show_relations=True)` likewise.
- Fields (`CorrelatedField`): `retarded` computes `K_R` natively (`compute_k`) then exports
  it to dense float64; `pauli_jordan` builds a dense **complex128** matrix for a purely
  imaginary antisymmetric operator; `wightman` uses `np.linalg.eigh` instead of the native
  skew eigensolver (`eig_skew`, R2_CATALOG); `entanglement_entropy` recomputes `K_R`,
  `iDelta` and `W` from scratch on every call. These return `np.ndarray` by design today.
- `synthetic`: every generator builds a dense n x n NumPy order, and `transitive_percolation`
  / `random_dag_order` / `poset` take the transitive closure with an O(n^3) NumPy loop.
- Sprinkling into Python spacetimes (`_sprinkle_python`, used by de Sitter, FLRW,
  Schwarzschild, the composition decorators and custom spacetimes) builds a dense n x n
  boolean batch and then sets bits one at a time from Python; without `is_causal_batch` it
  calls `is_causal` once per pair (de Sitter: 26 s at n = 2,000).

Not affected (genuine outputs): `pycauset.to_numpy(...)`, `CausalSet.coordinates()` (n x d),
and per-point coordinate math in `spacetime.py`.

**Reproduction**:
```python
import time, pycauset as pc
c = pc.CausalSet(n=20_000, seed=1)       # 2 s
t = time.time(); c.relation_fraction(); print(time.time() - t)   # ~105 s

pc.set_export_max_bytes(1)               # forbid dense NumPy export
c.past(0)                                # RuntimeError: Export to NumPy exceeds ...
```

**Root Cause**:
`TriangularBitMatrix` exposes only per-element `get`/`set` to Python: no relation count
(popcount), no row/column access as index arrays or bitsets, no native transitive reduction,
closure, or longest-chain/layering. The physics layer was therefore written against NumPy,
and the test suite only uses small n, so nothing caught it.

**Fix** (planned, R2 node `R2_NATIVE`):
Add native bitset primitives and route the physics layer through them: relation count via
popcount; `past`/`future` as column/row extraction; `links` via bit matmul (the popcount
kernel already exists); `longest_chain`/`layers` as a DP over rows; transitive closure and
validation natively; plots extract only the sampled submatrix; fields keep `K_R`/`iDelta`
as engine matrices (antisymmetric real storage, scalar `i`) and use the native eigensolvers,
with `W` cached. `tests/python/test_no_dense_fallback.py` holds a strict-xfail case per
operation; remove the marker as each one moves into the engine.

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


