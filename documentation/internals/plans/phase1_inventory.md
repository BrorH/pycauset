# R1_NUMPY - Phase 1 Inventory Report

**Date:** Jan 6, 2026
**Status:** Complete

## 1. Surface Map (Existing Interop)

### 1.1 Python Type Support
The following types currently expose `__array__` or the Buffer Protocol:

| PyCauset Type | Native C++ Type | NumPy Dtype | Buffer Proto? | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `FloatMatrix` | `DenseMatrix<double>` | `float64` | ✅ Yes | |
| `Float32Matrix` | `DenseMatrix<float>` | `float32` | ✅ Yes | |
| `IntegerMatrix` | `DenseMatrix<int32_t>` | `int32` | ✅ Yes | |
| `Int16Matrix` | `DenseMatrix<int16_t>` | `int16` | ✅ Yes | |
| `DenseBitMatrix` | `DenseBitMatrix` | `bool` | ❌ No | Exposes via `__array__` (converts to bool array). |
| `FloatVector` | `DenseVector<double>` | `float64` | ✅ Yes | |
| `BitVector` | `DenseVector<bool>` | `bool` | ✅ Yes | |
| `LazyMatrix` | `MatrixExpressionWrapper` | N/A | ❌ **MISSING** | `np.array(expr)` fails to evaluate. |

### 1.2 Import Rules (NumPy -> PyCauset)
Defined in `bind_vector.cpp` and `bind_matrix.cpp`:

- **Vectors (1D):**
    - `float32` -> **Promotes to `float64`** (Legacy behavior, see `bind_vector.cpp:250`).
    - `float64` -> `float64`.
    - `int32/64` -> `int32/64`.
    - `bool` -> `bool`.
- **Matrices (2D):**
    - `float32` -> `float32` (Preserved).
    - `float64` -> `float64`.
    - `int32` -> `int32`.

## 2. Identified Gaps & Bugs

### 2.1 Critical: Lazy Evaluation Interop (Gap)
**Symptom:** `np.array(A + B)` returns a 0-D object array wrapping the expression.
**Cause:** `MatrixExpressionWrapper` lacks `__array__` hooks.
**Impact:** Breaks "NumPy-like" feel; users must manually call `.eval()` which is unpythonic.
**Fix Required:** Bind `__array__` in `bind_expression.cpp` to trigger `eval_into` and return the result.

### 2.2 Critical: Snapshot Export Zero-Data (Bug)
**Symptom:** `np.array(pc.load("snap.pycauset"))` returns all zeros.
**Cause:** `MemoryMapper` (or `persistence.py`) alignment mismatch.
- `MemoryMapper` correctly skips the 64-byte header for `.tmp` files.
- For `.pycauset` snapshots involving a 4KB header/metadata block, the offset passed to `MemoryMapper` seems to be 0 (pointing to the header) instead of 4096 (pointing to payload).
**Fix Required:** Ensure `FileBackedMatrix` constructor receives the correct absolute payload offset when loading from a snapshot.

### 2.3 Import Performance (Gap)
**Symptom:** `import_matrix` uses element-wise copying loops in `bind_matrix.cpp`.
**Optimization:** `R1_PERF` introduced `MemoryGovernor::should_use_direct_path`. Import logic should check this and use `std::memcpy` (parallelized) when RAM permits.

### 2.4 Ergonomics (Gap)
- `IntVector` alias is missing (users expect `pc.vector(...)` factory, but direct type usage fails).
- `np.matmul(pycauset, pycauset)` likely fails or falls back to slow path (untested but `__array_ufunc__` logic for expressions is minimal).

## 3. Plan Update
Phase 2 (Correctness) must prioritize fixing **2.1** and **2.2** before adding new tests.
