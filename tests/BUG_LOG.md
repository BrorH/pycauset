# Bug Log

This document tracks all bugs discovered during development and testing. AI agents must append new findings to this file.

## [2025-12-03] Legacy Header Format Persistence

**Status**: Fixed
**Severity**: Critical
**Component**: Storage (C++ / Python)

**Description**:
The C++ `MemoryMapper` and `TriangularBitMatrix` classes were expecting a legacy 4096-byte header allocation, whereas the new Python `__init__.py` logic was using a ZIP-based format without this header. This caused memory misalignment and data corruption.

**Reproduction**:
Saving a matrix in Python and attempting to load it resulted in offset errors or garbage data.

**Root Cause**:
Incomplete refactoring of the C++ storage layer to match the new ZIP container specification.

**Fix**:
Removed all references to `HEADER_SIZE` and 4096-byte offsets in C++. Updated `MemoryMapper` to map exactly the data region specified by the ZIP directory.

---

## [2025-12-03] DenseVector Naming Inconsistency

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings

**Description**:
The Python `__init__.py` referred to `DenseVector` when the C++ binding exposed it as `FloatVector`.

**Reproduction**:
`from pycauset import DenseVector` failed or internal instantiation failed.

**Root Cause**:
Naming mismatch between C++ `py::class_` binding and Python wrapper logic.

**Fix**:
Updated `__init__.py` to use `FloatVector` (aliased correctly).

---

## [2025-12-03] Scalar Property Access Mismatch

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings

**Description**:
Python code attempted to call `obj.set_scalar(val)`, but the binding only exposed a property `.scalar`.

**Reproduction**:
`matrix.set_scalar(2.0)` raised `AttributeError`.

**Root Cause**:
`def_property` in pybind11 creates a property, not a setter method.

**Fix**:
Changed Python code to use `obj.scalar = val`.

---

## [2025-12-03] CausalSet Saving Logic Missing

**Status**: Fixed
**Severity**: High
**Component**: Python Wrapper

**Description**:
`CausalSet` objects (wrappers around `TriangularBitMatrix`) could not be saved using `pycauset.save()` because the dispatcher didn't recognize the type.

**Reproduction**:
`pycauset.save(causal_set, "file.pycauset")` failed.

**Root Cause**:
Missing `isinstance` check for `CausalSet` in the save function.

**Fix**:
Added logic to extract the underlying matrix from `CausalSet` before saving.

---

## [2025-12-03] Windows Unicode Temp File Deletion

**Status**: Fixed
**Severity**: Low
**Component**: C++ Storage

**Description**:
Temporary files with Unicode characters in their path (e.g., Chinese characters) were not being deleted on Windows.

**Reproduction**:
Creating a temp matrix in a folder named "测试" and closing it left the file on disk.

**Root Cause**:
`std::filesystem::path` on Windows requires wide strings or UTF-8 `char8_t` casting for non-ASCII paths. Standard `std::string` was failing.

**Fix**:
Added `reinterpret_cast<const char8_t*>` when constructing `std::filesystem::path` from the filename string in `PersistentObject::close()`.

---

## [2025-12-03] Persistence Loading Offset Error

**Status**: Fixed
**Severity**: Critical
**Component**: Python `__init__.py`

**Description**:
Loading a matrix from a ZIP file resulted in an empty (all-zero) matrix, despite data existing on disk.

**Reproduction**:
`m = pycauset.load("data.pycauset"); m.get(0, 1)` returned `False` even if it should be `True`.

**Root Cause**:
A monkey-patch in `__init__.py` (`_patched_triangular_bit_matrix_init`) was intercepting the internal constructor call used by `load()`. It stripped the `offset` argument (treating it as a user-provided path argument to be ignored), forcing the C++ object to initialize with `offset=0`.

**Fix**:
Updated the patch to detect the 6-argument constructor signature used by `load()` and pass arguments through unchanged.

---

## [2025-12-03] Missing Scalar Addition

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings / C++ Core

**Description**:
In-place addition of a scalar to a matrix (`A += 5.0`) raises `TypeError`. The bindings do not support scalar addition, only matrix addition.

**Reproduction**:
```python
m = pycauset.FloatMatrix(5)
m += 5.0  # Raises TypeError
```

**Root Cause**:
`MatrixBase` and subclasses lacked `add_scalar` method, and `bindings.cpp` `__add__` did not handle scalar types.

**Fix**:
Implemented `add_scalar` in `MatrixBase` (virtual), `DenseMatrix`, `TriangularMatrix`, `DenseBitMatrix`, `TriangularBitMatrix`, and `IdentityMatrix`. Updated `bindings.cpp` to call `add_scalar` when operand is int or float.

---

## [2025-12-03] Strict BitMatrix Assignment

**Status**: Open (Design Choice)
**Severity**: Low
**Component**: Python Bindings

**Description**:
Assigning a non-0/1 integer to a `BitMatrix` element (`m[0,0] = 5`) raises `TypeError: Integer assignments must be 0 or 1`. It does not implicitly cast to boolean `True`.

**Reproduction**:
```python
m = pycauset.DenseBitMatrix(5)
m[0, 0] = 5  # Raises TypeError
```

**Root Cause**:
Explicit validation in `bindings.cpp` lambda for `set`.

**Fix**:
None. Tests updated to handle `TypeError`.

---

## [2025-12-03] Triangular Set Warning vs Error

**Status**: Open (Design Choice)
**Severity**: Low
**Component**: Python Bindings

**Description**:
Setting a value in the lower triangle or diagonal of a `TriangularMatrix` emits a `UserWarning` and ignores the operation, rather than raising an exception.

**Reproduction**:
```python
m = pycauset.TriangularBitMatrix(5)
m[1, 0] = True  # Emits UserWarning
```

**Root Cause**:
Implementation choice in `TriangularMatrix::set` (or python wrapper) to warn instead of throw.

**Fix**:
None. Tests updated to catch warning.

---

## [2025-12-03] Vector Transpose NumPy Shape Mismatch

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings

**Description**:
Transposed vectors (`v.T`) were being converted to NumPy arrays with shape `(N,)` instead of `(1, N)`. This caused dimension mismatch errors in linear algebra operations expecting row vectors.

**Reproduction**:
```python
v = FloatVector(3)
vt = v.T
assert np.array(vt).shape == (1, 3) # Failed, was (3,)
```

**Root Cause**:
The `__array__` implementation in `bindings.cpp` did not check the `is_transposed()` flag and always returned a 1D array.

**Fix**:
Updated `__array__` lambda in `bindings.cpp` to check `is_transposed()` and return a reshaped array `(1, N)` if true.

---

## [2025-12-03] Mixed Matrix Multiplication Runtime Error

**Status**: Fixed
**Severity**: High
**Component**: C++ Matrix Operations

**Description**:
Multiplying an `IdentityMatrix` by a `DenseMatrix` (converted from NumPy) caused a `RuntimeError: Mixed Dense/Triangular multiplication not yet optimized`.

**Reproduction**:
```python
id_mat = IdentityMatrix(3)
np_mat = np.eye(3)
res = id_mat @ np_mat # Failed
```

**Root Cause**:
The `dispatch_matmul` function in `bindings.cpp` lacked a specific case for `IdentityMatrix` interacting with generic `MatrixBase` types, falling through to the unoptimized/unsupported mixed type error.

**Fix**:
Added explicit dispatch logic in `bindings.cpp` to handle `IdentityMatrix` multiplication by delegating to `multiply_scalar` on the other operand.

---

## [2025-12-03] Missing `_native.load` in `compute_k`

**Status**: Fixed
**Severity**: High
**Component**: Python Wrapper

**Description**:
The `compute_k` function in `__init__.py` attempted to call `_native.load`, which does not exist (loading is handled by `MatrixFactory` or the Python `load` wrapper).

**Reproduction**:
Calling `pycauset.field.propagator()` or `compute_k()` raised `AttributeError`.

**Root Cause**:
Refactoring left a call to a non-existent native function.

**Fix**:
Refactored `compute_k_matrix` in C++ to return the result object directly, and updated Python `compute_k` to use this return value instead of attempting to load from disk.

---

## [2025-12-03] CausalSet Loading Attribute Error

**Status**: Fixed
**Severity**: High
**Component**: Python Wrapper

**Description**:
Loading a saved `CausalSet` resulted in an object that was missing the `n` attribute and other `CausalSet` properties, because `pycauset.load` was returning the raw matrix instead of reconstructing the `CausalSet` object.

**Reproduction**:
```python
c = CausalSet(n=10)
c.save("test.causet")
c_loaded = pycauset.load("test.causet")
print(c_loaded.n) # AttributeError
```

**Root Cause**:
`pycauset.load` did not check the `object_type` metadata field to reconstruct the high-level `CausalSet` wrapper.

**Fix**:
Updated `pycauset.load` to check `metadata["object_type"] == "CausalSet"` and reconstruct the object using the stored spacetime arguments and matrix.

---

## [2025-12-03] Vector Transpose Persistence Failure

**Status**: Fixed
**Severity**: Medium
**Component**: Storage / Bindings

**Description**:
Saving and loading a transposed vector resulted in a non-transposed vector.

**Reproduction**:
```python
v = FloatVector(3)
vt = v.T
vt.save("vec.pycauset")
vt_loaded = pycauset.load("vec.pycauset")
assert vt_loaded.is_transposed() # Failed
```

**Root Cause**:
1. `VectorBase` did not have the `save` method monkey-patched.
2. `PersistentObject` did not expose `is_transposed` to Python, preventing the save logic from reading the state correctly.

**Fix**:
1. Exposed `is_transposed` in `bindings.cpp`.
2. Monkey-patched `VectorBase.save` in `__init__.py`.
3. Updated `pycauset.load` to apply `set_transposed(True)` if metadata indicates it.


## [2025-12-03] Temporary File Leak

**Status**: Fixed
**Severity**: High
**Component**: Storage / Resource Management

**Description**:
Hundreds of temporary files (.pycauset and .bin) were accumulating in the .pycauset directory and not being deleted upon object destruction or process exit.

**Reproduction**:
Run any script that creates temporary matrices (e.g. intermediate results of operations) and observe the .pycauset directory growing indefinitely.

**Root Cause**:
The default constructor of PersistentObject (used by MatrixBase) did not initialize the is_temporary_ member variable. As a result, it held a garbage value (often false), causing the close() method to skip file deletion even for objects that were explicitly marked as temporary in Python.

**Fix**:
Updated PersistentObject::PersistentObject() to explicitly initialize is_temporary_ to false. This ensures that the flag is only true when explicitly set via set_temporary(true), which is done by the Python wrapper for auto-generated files.


## [2025-12-03] ComplexMatrix Missing Scalar Operations

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings / Matrix Operations

**Description**:
`ComplexMatrix` supports matrix-matrix addition and multiplication but fails when operated with a scalar (real or complex). Both `__add__` and `__mul__` raise `TypeError` when a scalar is passed.

**Reproduction**:
```python
import pycauset
cm = pycauset.ComplexMatrix(2)
cm.set(0, 0, 1.0 + 1.0j)

# Fails with TypeError
cm_scaled = cm * (2.0 + 3.0j) 
cm_added = cm + (1.0 + 0.0j)
```

**Root Cause**:
The `bindings.cpp` definition for `ComplexMatrix` only binds `__add__` and `__mul__` for `ComplexMatrix` arguments. Overloads for `std::complex<double>` or `double` are missing.

**Fix**:
Implemented `multiply_scalar` and `add_scalar` in `include/ComplexMatrix.hpp` which perform element-wise operations on the real and imaginary parts. Updated `src/bindings.cpp` to expose `__mul__`, `__rmul__`, `__add__`, and `__radd__` overloads accepting `std::complex<double>`.

---

## [2025-12-04] Skew Solver Returns Excess Eigenvalues

**Status**: Fixed
**Severity**: Medium
**Component**: Eigenvalue Solver

**Description**:
The `eigvals_skew` function returned all eigenvalues found in the Krylov subspace (size $m \ge k$) instead of the requested top $k$. This caused test failures where `evals.size() != k`.

**Reproduction**:
```python
N = 5
k = 2
evals = pycauset.eigvals_skew(matrix, k)
assert evals.size() == k # Failed, returned 4 or 6
```

**Root Cause**:
The solver returned the full spectrum of the projected matrix $T$ without filtering.

**Fix**:
Modified `src/Eigen.cpp` to sort the eigenvalues by magnitude and truncate the result vector to size $k$ before returning.

---

## [2025-12-04] ComplexVector Not Iterable in Python

**Status**: Fixed
**Severity**: Medium
**Component**: Python Bindings

**Description**:
`ComplexVector` could not be iterated over in Python (e.g., `[x for x in vec]`), raising `TypeError`.

**Reproduction**:
```python
v = pycauset.eigvals_skew(A, k=10)
l = list(v) # TypeError: 'ComplexVector' object is not iterable
```

**Root Cause**:
Missing `__iter__` or `__getitem__` binding for iteration support.

**Fix**:
Added `__len__` and `__getitem__` to `ComplexVector` bindings in `src/bindings.cpp`.

