# R1_NUMPY — Phase 1 Surface Map (Inventory)

**Date:** 2025-12-28

This document is a factual map of the current NumPy interop surface in the repo.
It exists to support Phase 2 (tests) and to prevent “optimizing the wrong thing”.

---

## 1) Public entrypoints (Python)

**Import (NumPy → PyCauset)**

- `pycauset.vector(source, dtype=None, *, max_in_ram_bytes=None, **kwargs)`
  - Implementation: `python/pycauset/__init__.py` → `_internal/factories.py::vector_factory(...)`
- `pycauset.matrix(source, dtype=None, **kwargs)`
  - Implementation: `python/pycauset/__init__.py` → `_internal/matrix_api.py::Matrix.__new__(...)` and factory helpers

**Export (PyCauset → NumPy)**

- `np.asarray(obj)` / `np.array(obj)`
  - Uses `obj.__array__` (NumPy array protocol)
- `pycauset.to_numpy(obj, *, allow_huge=False, dtype=None, copy=True)`
  - Implementation: `python/pycauset/__init__.py` → `_internal/export_guard.py::export_to_numpy(...)`
- `pycauset.set_export_max_bytes(limit: int | None)`
  - Implementation: `python/pycauset/__init__.py` → `_internal/export_guard.py::set_max_bytes(...)`

**On-disk conversion helper**

- `pycauset.convert_file(src_path, dst_path, *, dst_format=None, allow_huge=False, dtype=None, npz_key=None)`
  - Implementation: `python/pycauset/__init__.py` (uses `to_numpy(...)` for exports)

---

## 2) Python-visible participating types

### 2.1 Native extension base types

- `pycauset._pycauset.MatrixBase`
  - Binding: `src/bindings/bind_matrix.cpp`
  - Properties used by export guard:
    - `get_backing_file()` / `backing_file`
    - `rows()`, `cols()` and/or `shape`
    - `get(i, j)`
- `pycauset._pycauset.VectorBase`
  - Binding: `src/bindings/bind_vector.cpp`
  - Properties used by export guard:
    - `get_backing_file()` / `backing_file`
    - `size()` and/or `shape`
    - `get(i)`

Concrete native classes are registered under these bases (examples):

- Matrices: `FloatMatrix`, `Float32Matrix`, `IntegerMatrix`, `DenseBitMatrix`, `TriangularBitMatrix`, `ComplexFloat32Matrix`, `ComplexFloat64Matrix`, …
- Vectors: `FloatVector`, `Float32Vector`, `IntegerVector`, `BitVector`, `ComplexFloat32Vector`, `ComplexFloat64Vector`, …

### 2.2 Python-only “composite” types

- `_internal/blockmatrix.py::BlockMatrix`
  - Implements `__array__` as a debug/fallback path, delegating to `_internal/export_guard.export_to_numpy(..., allow_huge=False)`.

---

## 3) NumPy protocols implemented (current)

### 3.1 `__array__`

- C++ bindings provide `__array__(dtype, copy)` on `MatrixBase` and `VectorBase`.
- At import time, the Python facade overrides the base-class `__array__` with a guarded implementation:
  - `python/pycauset/__init__.py`:
    - `_MatrixBaseType.__array__ = _guarded_array_export`
    - `_VectorBaseType.__array__ = _guarded_array_export`

So the effective `np.array(obj)` route for native matrices/vectors is:

1) NumPy calls `obj.__array__(dtype=..., copy=...)`
2) `_guarded_array_export` delegates to `_internal/export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=copy_flag)`
3) Export guard enforces the materialization safety boundary.

### 3.2 `__array_priority__`

- Python facade sets:
  - `_MatrixBaseType.__array_priority__ = 1e6`
  - `_VectorBaseType.__array_priority__ = 1e6`

This influences mixed-operand dispatch (NumPy vs PyCauset operator precedence).

### 3.3 `__array_ufunc__` / `__array_function__`

- Not implemented for the native matrix/vector types today.
- Current behavior for NumPy ufuncs / `np.*` functions is typically:
  - NumPy coerces the PyCauset operand(s) to `ndarray` (calling `__array__`), then proceeds.

This is why the export safety guard is critical.

---

## 4) Import paths (NumPy → PyCauset)

### 4.1 `pycauset.vector(np.ndarray)`

Implementation: `_internal/factories.py::vector_factory`

- Fast-path for supported numeric/bool dtypes uses `native.asarray(data)`.
- Complex vectors are constructed via the dedicated complex vector classes instead of `native.asarray`:
  - complex64 → `ComplexFloat32Vector(np_array)`
  - complex128 → `ComplexFloat64Vector(np_array)`

Rationale (current state): `native.asarray` supports complex for 2D but not for 1D; using the concrete vector constructors avoids a hard failure.

### 4.2 `pycauset.matrix(np.ndarray)`

Implementation: `_internal/matrix_api.py::Matrix.__new__`

- For supported dtypes and when `native.asarray` is available, the matrix constructor routes directly to `native.asarray(data)`.
- For other inputs, it falls back to list coercion and then constructs/fills native matrices.

### 4.3 Rank constraints

- `pycauset.vector(...)` rejects non-1D inputs.
- `pycauset.matrix(...)` is a convenience factory:
  - 1D input → vector
  - 2D input → matrix
  - Scalars / 0D are rejected (`_is_scalar_0d`).

---

## 5) Export guard (materialization safety)

Implementation: `python/pycauset/_internal/export_guard.py`

### 5.1 Backing classification

- `backing_kind(obj)` inspects `get_backing_file` / `backing_file`.
- Special case:
  - `":memory:"` is treated as in-RAM.
- Heuristic kinds:
  - `.pycauset` → `snapshot`
  - `.tmp` / `.raw_tmp` → temp/spill-like

### 5.2 Safety rule (current)

- `ensure_export_allowed(obj, allow_huge=False, ceiling_bytes=...)`:
  - If object is file-backed/out-of-core (kind is set and not `snapshot`): export is blocked unless `allow_huge=True`.
  - If `ceiling_bytes` is set and estimated bytes exceed it: export is blocked.

### 5.3 Size estimation

- `estimate_materialized_bytes(obj)` uses:
  - shape (`rows/cols` or `shape`), and
  - dtype token (`obj.dtype` or type name tokenization)
- Bit storage is treated as 1/8 byte per element.

---

## 6) Export implementation (`export_to_numpy`)

Implementation: `python/pycauset/_internal/export_guard.py::export_to_numpy`

Order of operations:

1) Enforce guardrail (`ensure_export_allowed`).
2) Choose output dtype (`_infer_numpy_dtype`).
3) Attempt a `memoryview(obj)` materialization path.
4) Otherwise allocate `np.empty(...)` and fill via `get(i[, j])` when available.
5) Final fallback: `np.array(obj, dtype=..., copy=...)`.

Notes:

- The `copy=` argument is currently not a true zero-copy view path; export is generally materializing a new NumPy buffer.
- Vector exports handle both column-vector and transposed row-vector shapes (e.g. `FloatVector.T` with shape `(1, N)`), materializing via single-index `get(i)`.
- Complex dtype mapping promotes complex16-like tokens to NumPy `complex64` (NumPy has no complex16).

---

## 7) Known gaps / follow-ups for Phase 2 tests

- Done: regression tests for `pycauset.vector(np.array(..., dtype=np.complex64/complex128))` are in `tests/python/test_numpy_vector_complex_import.py`.
- Add/confirm tests for the export guard boundary:
  - RAM (`:memory:`) allowed
  - snapshot (`.pycauset`) allowed
  - spill/temp (`.tmp`) blocked unless `allow_huge=True`
  - ceiling enforcement via `set_export_max_bytes`

---

## 8) Design-chief questions (only if you want to change behavior)

1) `native.asarray` currently promotes **float32 1D** vectors to float64 vectors (while float32 2D matrices remain float32). Is that intentional long-term, or do you want float32 vectors preserved as float32?
