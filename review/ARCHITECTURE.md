# PyCauset Architecture Map

**What this is:** the "open the hood" map. Every moving part of the system, what it
owns, and how a call travels from the user's Python down to disk and back. Read this
before touching code. It describes the code **as it is** (as of 2026-11-24, commit
`046c5ab`), not as the older design docs say it should be.

**Who maintains it:** whoever changes a subsystem's responsibilities or wiring updates
this file in the same commit. If you move a file, fix its path here.

**Coverage:** per-file executed-line numbers are in `review/deadcode/`; they show how
much of each file is actually alive.

Status markers used below: **[OK]** coherent, **[SMELL]** works but is patchwork,
**[DEAD]** unreferenced, **[BUG]** see `FINDINGS.md`.

---

## 1. The 30-second version

PyCauset is two products sharing one engine:

1. **The engine**: "NumPy for causal sets". Matrices and vectors of 14 dtypes and
   ~8 structures (dense, triangular, bit-packed, symmetric, diagonal, identity, ...)
   that live in RAM or transparently spill to memory-mapped files on disk.
2. **The physics suite**: causal sets, spacetimes, sprinkling, fields and
   propagators, and visualization. It is built *on top of* the engine. A `CausalSet`
   is a wrapper around a bit-packed upper-triangular matrix.

Stack, top to bottom:

```
 user code:  pc.CausalSet(...), pc.matrix(...), A @ B, pc.invert(A), pc.save(...)
 ─────────────────────────────────────────────────────────────────────────────────
 PHYSICS (pure Python)     causet.py  spacetime.py  field.py  synthetic.py  vis.py
 ─────────────────────────────────────────────────────────────────────────────────
 PYTHON FACADE             __init__.py (2416 lines): public names, factories,
                           dtype tokens, debug mode; delegates into _internal/
 PYTHON GLUE (_internal)   ops.py          linalg routing, NumPy fallbacks
                           properties.py   property system + operator wrapping
                           patching.py     operator wrapping (temp-file tracking)
                           persistence.py  .pycauset container read/write
                           export_guard.py NumPy export size guard
                           blockmatrix.py  Python-level block matrices
                           ... (30 modules, see INVENTORY.md)
 ─────────────────────────────────────────────────────────────────────────────────
 PYBIND11 BINDINGS         src/bindings.cpp + src/bindings/*.cpp  -> _pycauset.so
 ─────────────────────────────────────────────────────────────────────────────────
 C++ ENGINE (libpycauset_core.so)
   types        include/pycauset/matrix/*, vector/*      MatrixBase/VectorBase tree
   math entry   src/math/LinearAlgebra.cpp               pycauset::add/matmul/...
   dispatch     ComputeContext -> AutoSolver -> CpuDevice -> CpuSolver (or CudaDevice)
   streaming    StreamingManager + ComputeWorker         tiled out-of-core loops
   storage      PersistentObject -> MemoryMapper         RAM buffer or mmap file
                MemoryGovernor                           RAM budget, LRU spill
                IOAccelerator                            madvise/prefetch hints
   physics      Sprinkler.cpp + causet/Spacetime.hpp     native Minkowski sprinkling
 ─────────────────────────────────────────────────────────────────────────────────
 BLAS/LAPACK: OpenBLAS + LAPACKE (system or fetched); Eigen 3.4 (fetched); OpenMP
 OPTIONAL: pycauset_cuda.so + _pycauset_cuda.so (cuBLAS/cuSOLVER). ON HOLD.
```

---

## 2. C++ engine

### 2.1 Object model [OK core, SMELL edges]

```
PersistentObject            (global namespace, include/pycauset/core/PersistentObject.hpp)
 │  owns: shared_ptr<MemoryMapper> mapper_   <- the actual bytes (RAM or mmap)
 │  view metadata (makes transpose/scale/conj O(1), no copy):
 │     scalar_ (complex<double>), is_transposed_, is_conjugated_, seed_
 │  properties_flags_ (bitmask mirror of Python properties, PropertyFlags.hpp)
 │  storage_state_ RAM_ONLY | DISK_BACKED | TRANSITIONING
 │  CoW: ensure_unique() deep-copies shared storage before a write
 │
 ├── MatrixBase  (matrix/MatrixBase.hpp)  get/set_element_as_double/complex,
 │     │         multiply_scalar, add_scalar, transpose, write_block, ...
 │     ├── DenseMatrix<T>          T in {int8..int64, uint8..uint64, float16/32/64,
 │     │                            complex<float>, complex<double>}
 │     ├── DenseMatrix<bool>       = DenseBitMatrix (bit-packed)
 │     ├── TriangularMatrixBase
 │     │     ├── TriangularMatrix<T>      (TriangularFloatMatrix, TriangularIntegerMatrix)
 │     │     ├── TriangularMatrix<bool>   = TriangularBitMatrix  <- THE causal matrix
 │     │     └── SymmetricMatrix<T> ── AntiSymmetricMatrix<T> (packed upper triangle)
 │     ├── DiagonalMatrix<T>, IdentityMatrix<T>  (O(n) / O(1) storage)
 │     └── ComplexFloat16Matrix    (hand-written: two float16 planes)
 └── VectorBase   DenseVector<T>, DenseVector<bool> (BitVector), UnitVector,
                  ComplexFloat16Vector
```

* `Types.hpp` has two enums: `DataType` (16 dtypes) and `MatrixType`. **[SMELL]**
  `MatrixType` mixes *structure* (CAUSAL, TRIANGULAR_FLOAT, IDENTITY, SYMMETRIC,
  DIAGONAL) with *dtype* (INTEGER, DENSE_FLOAT), so structure × dtype is encoded
  inconsistently.
* Python sees ~41 concrete classes (one per structure × dtype), named
  inconsistently: `FloatMatrix` = float64 and `IntegerMatrix` = int32, but
  `Float32Matrix`, `Int8Matrix`, `UInt64Matrix`, and so on.
* **Invariant (F-001):** integer and bit matrices never carry a lazy `scalar_`; scalar ops
  on them are computed by `ScalarOps` with NumPy result dtypes. Float/complex matrices may
  carry a lazy scalar, and every reader of them must apply it.
* **[F-032]** Python classes exist only for some template instantiations
  (Triangular{bit,int32,float64}, Diagonal/Identity{float64}); others reach Python as a
  bare `MatrixBase`.

### 2.2 Construction: ObjectFactory [OK]
`core/ObjectFactory.cpp` creates matrices and vectors from `(rows, cols, DataType,
MatrixType)` and re-creates them on load from explicit metadata. Result types for
binary ops come from `MatrixTypeResolver.hpp` (structure) and `PromotionResolver.cpp`
(dtype promotion, including the float-underpromotion warning). `StructureResolver.hpp`
does structure algebra (e.g. diag ∘ X → diag).

### 2.3 Operation entry and dispatch
```
pybind lambda (bind_matrix.cpp)
   └─> pycauset::add / matmul / ...         src/math/LinearAlgebra.cpp  (allocates result
        │                                    via ObjectFactory, picks structure/dtype)
        └─> ComputeContext::instance()       singleton; owns an AutoSolver
             └─> AutoSolver                  src/compute/AutoSolver.cpp: routes CPU vs GPU
                  │                          by size and HardwareProfile (GPU on hold)
                  ├─> CpuDevice              thin adapter
                  │    └─> CpuSolver         src/compute/cpu/CpuSolver.cpp, 5,867 lines:
                  │                          ALL CPU kernels (matmul, elementwise, vector ops,
                  │                          LAPACK wrappers, eigensolvers, SIMD fast paths)
                  └─> CudaDevice             src/accelerators/cuda/ (ON HOLD)
```
* `ComputeDevice.hpp` is a fat interface: ~40 pure virtuals, one per op.
* **[SMELL]** `CpuSolver.cpp` is a monolith. `matmul` alone is ~880 lines and
  `matrix_vector_multiply` ~490. The vector ops repeat dtype-dispatch ladders
  (~250 lines each).
* `OpRegistry` (`core/OpRegistry.hpp`, `src/core/OpRegistration.cpp`) is a
  per-op "support contract" table, exposed to Python for introspection. It does not
  drive dispatch.
* SIMD: no global `-march`. AVX2/AVX-512 kernels are per-function `target`
  attributes with cpuid runtime dispatch (see CMakeLists comment).

### 2.4 Out-of-core streaming
* C++: `compute/StreamingManager` plans tiles from MemoryGovernor budgets and runs
  them through a `ComputeWorker` (`cpu/CpuComputeWorker`). Only `CpuDevice` uses it.
  **[F-026] It is 0% executed** by the test suite and demos. In practice, disk-backed
  operands are processed through ordinary mmap paging by the normal kernels.
* Python: `_internal/streaming_manager.py` is a **separate** Python-side planner and
  observer (`last_io_trace`, IO traces). **[SMELL]** Two components with the same
  name and overlapping purpose.

### 2.5 Storage [OK design]
* `MemoryMapper` owns the bytes: an anonymous RAM buffer or an mmapped file
  (Windows and POSIX paths, pinning, offset mapping into a `.pycauset` container).
* `MemoryGovernor` is the global RAM budget plus an LRU of live objects. It calls
  `spill_to_disk()` on the least-recently-used objects under pressure. Reads
  `MemAvailable` on Linux.
* `IOAccelerator` turns `MemoryHint`s (sequential/strided "Lookahead Protocol")
  into OS prefetch and madvise calls.
* `StorageUtils` holds the global storage root and memory threshold. The default
  spill directory is `./.pycauset`, **relative to the current working directory**
  (`_internal/runtime.py`).
* Singletons: `MemoryGovernor`, `ComputeContext` and `OpRegistry` are leaked heap
  singletons (deliberate, for safe interpreter teardown). `ThreadPool` is a normal
  Meyers static. **[SMELL]** `TODO.md` says ThreadPool is leaked too; it isn't.

### 2.6 Threading **[BUG F-004, F-005]**
* `core/ParallelUtils.hpp`: a home-grown `ThreadPool` (hardware_concurrency
  workers) plus `ParallelFor` and `ParallelBlockMap`. About 200 call sites, almost
  all in CpuSolver.
* Also in play in the same process: OpenMP (`libgomp`), OpenBLAS's own pthread pool
  (bumped per-GEMM in `CpuSolver::matmul`), and NumPy's bundled OpenBLAS. That is
  **four independent thread pools**, an oversubscription risk.
* Nested `ParallelFor` deadlocks, and the exception path can use-after-free (see
  FINDINGS).

### 2.7 Lazy expressions
`include/pycauset/matrix/expression/*`: expression templates (`BinaryExpression`,
`UnaryExpression`, `ScalarExpression`, `MatrixRefExpression`) wrapped for Python as
`LazyMatrix` (`bind_expression.cpp`). Some float-scalar operations return a
`LazyMatrix` instead of a concrete matrix, and it does not support item assignment.
**[SMELL]**: users get different result types for `A*2.0` versus `A*2`.

### 2.8 Physics in C++
`causet/Spacetime.hpp` (native MinkowskiDiamond, Cylinder, Box) and `Sprinkler.cpp`
(seeded sprinkling that builds the TriangularBitMatrix directly, interruptible).


---

## 3. Bindings (`src/bindings/`, module `_pycauset`)

| file | lines | binds |
|---|---|---|
| `bindings.cpp` | small | `PYBIND11_MODULE`, calls the `bind_*` functions |
| `bind_core.cpp` | 273 | storage config, threads, memory threshold, IOAccelerator, hints |
| `bind_matrix.cpp` | **~4,700** | every matrix class, operators (~8 overloads per operator; scalar overloads delegate to ScalarOps for int/bit), NumPy buffer protocol and generic native export (`export_dense_any`), construction from NumPy |
| `bind_vector.cpp` | 1,371 | vector classes |
| `bind_complex.cpp` | 174 | complex16 extras |
| `bind_expression.cpp` | 260 | `LazyMatrix` |
| `bind_math.cpp` | | free functions (linalg) |
| `bind_causet.cpp` | | native spacetimes, `sprinkle`, `make_coordinates` |
| `bind_op_registry.cpp` | | OpRegistry introspection |
| `bind_cuda.cpp` | | separate `_pycauset_cuda` module (ON HOLD) |
| `binding_warnings.*`, `numpy_export_guard.hpp`, `bindings_common.hpp` | | helpers |

Operator semantics are split between C++ overloads here and Python wrappers in
`_internal/`, and neither side is complete. Unary minus (`__neg__`) is not defined
anywhere, and `int * Matrix` fails for most dtypes (F-002).

---

## 4. Python layer

### 4.1 Import sequence (`python/pycauset/__init__.py`)
1. Define dtype tokens (`pc.float64 = "float64"`, ...), debug-mode helpers.
2. `_internal/native.py`: Windows DLL search paths, import `_pycauset`.
3. `_internal/native_setup.py`: set OpenBLAS threads via ctypes, GPU install hint.
4. Import `field`, `spacetime`, `synthetic`, `causet`.
5. Alias native classes (`FloatMatrix = _native.FloatMatrix`, ...), many behind
   `getattr(_native, ..., None)` guards (73 of them).
6. **Monkeypatch native classes** (see 4.2).
7. Build `Runtime`, `IOObservability`, `StreamingManager` singletons, register
   atexit cleanup of temp files.
8. Define all public functions (factories, linalg, I/O, plotting shortcuts) and
   `__all__` (137 names). `__getattr__` hides internal native names.

### 4.2 The wrapper stack **[SMELL: central patchwork]**
At import, native pybind methods are replaced with Python closures, in layers:

```
A * B  ->  properties.py wrapper   (propagate is_symmetric/... to the result)
        ->  patching.py wrapper    (register temp backing file of the result)
        ->  pybind overload set    (~8 C++ overloads: matrix, ndarray, float, int,
                                    complex, numpy-scalar, fallback...)
        ->  C++
```
Measured depth: FloatMatrix and IntegerMatrix `__mul__/__add__/__sub__/__rmul__/
transpose` have 2 Python layers; TriangularBitMatrix has 1; `__matmul__` has 1
(the debug-mode timer). Different classes therefore get different semantics and
overheads. This is the most likely cause of the operator inconsistencies (F-001,
F-002) and of elementwise ops running at 0.2–0.3× NumPy speed.

### 4.3 `_internal/` responsibilities

| module | lines | role |
|---|---|---|
| `ops.py` | 1,691 | linalg front-ends: try native, else fall back to `numpy.linalg` (eigh, eigvalsh, svdvals, matrix_rank always use NumPy) |
| `persistence.py` | 1,664 | `.pycauset` container: header slots, metadata blocks, save/load, cached derived objects |
| `properties.py` | 1,199 | user-visible property dict (`is_symmetric`, `is_upper_triangular`, ...), propagation rules, operator wrappers |
| `blockmatrix.py` | 852 | Python BlockMatrix (tiles of native matrices) |
| `matrix_api.py` | 618 | `Matrix` factory shim; contains a leftover **pure-Python list-backed matrix** reached only from `pc.matrix`/`zeros` fallback branches that never fire (DEADCODE Tier B); empty `TriangularMatrix` shim exported as `pc.TriangularMatrix` |
| `io_observability.py` | 335 | IO trace records (`last_io_trace`) |
| `export_guard.py` | 319 | refuse NumPy export of file-backed or huge objects |
| `streaming_manager.py` | 288 | Python streaming planner (see 2.4) |
| `factories.py` | 247 | `zeros/ones/empty` helpers |
| `lazy_allocation.py` | 200 | dtype-deferred `zeros()` with no dtype |
| `dtypes.py` | 167 | dtype normalization |
| `big_blob_cache.py`, `linalg_cache.py` | | cache derived results (inverse, eigenvalues) inside containers |
| `support_matrix.py` | 148 | executable dtype × op support table (tests/tools only) |
| `submatrix_view.py`, `dense_python.py`, `thunks.py` | | block/view helpers |
| `runtime.py`, `native.py`, `native_setup.py`, `native_threshold.py` | | process setup |
| `formatting.py`, `coercion.py`, `warnings.py`, `gpu_hint.py` | | small utilities |
| `patching.py` | 102 | operator wrappers (temp-file tracking) |
| `storage_debug.py` | 97 | debug tool, used only by its own test |

Exception handling: the Python package has 257 broad `except`/`except Exception`
clauses, 126 of which silently `pass`. Many failure modes are invisible by design.

### 4.4 Physics suite (pure Python over the engine)
* `causet.py`: `CausalSet(n, spacetime, seed)`. For native spacetimes it calls
  `_native.sprinkle` (C++). Otherwise it samples in Python via `Spacetime.sample` /
  `is_causal`. Holds the TriangularBitMatrix. Methods: links, past/future, interval,
  chains, layers, Myrheim–Meyer dimension, plots, save/load.
* `spacetime.py`: `Spacetime` ABC plus a name registry. Minkowski Diamond, Cylinder
  and Box exist **twice**, as Python classes and as native C++ mirrors. Also
  Transformed, Restricted, Conformal and Periodic wrappers, a `create()` recipe
  builder, `export_python`, and curved DeSitter, AntiDeSitter (raises), FLRW and
  Schwarzschild.
* `field.py`: `pc.field("scalar", mass=...)` → `Field.on(causet)` → retarded and
  advanced propagators, Pauli–Jordan, Sorkin–Johnston `State`.
* `synthetic.py`: chain, antichain, transitive percolation, random DAG, product
  order, poset.
* `vis.py`: Plotly embedding, Hasse diagram, causal-matrix heatmap.

---

## 5. Representative call flows

**`C = pc.CausalSet(n=3000, seed=42)`** → causet.py → `MinkowskiDiamond(2)` (Python)
→ its native mirror → `_native.sprinkle(st, n, seed)` → Sprinkler.cpp sprinkles,
sorts by time, fills the `TriangularBitMatrix` bits → MemoryGovernor decides RAM or
disk.

**`A * 3` / `A - 1` / `-A` (any scalar op)** → pybind overload (typed or catch-all;
`to_scalar_arg` classifies Python/NumPy numbers) → integer/bit matrices, and any case
where the result dtype changes, go to `pycauset::scalar_op` (`src/math/ScalarOps.cpp`):
NumPy result dtypes, exact typed kernel, triangular kept when 0 ↦ 0. Same-dtype float ops
keep the O(1) lazy-scalar/LazyMatrix paths.

**`A @ B` (float64 dense)** → Python debug wrapper → pybind `__matmul__` →
`pycauset::matmul` → ComputeContext → AutoSolver → CpuDevice → `CpuSolver::matmul`
→ OpenBLAS `cblas_dgemm` (thread count bumped then restored). If an operand is
disk-backed and the result exceeds budget → StreamingManager tiles it.

**`pc.eigh(A)`** → ops.py → **`numpy.linalg.eigh`** on a full in-RAM copy (native
`CpuSolver::eigh` exists but is not the path taken). It cannot run out-of-core.

**`pc.eig(A)`** → ops.py → native `pycauset::eig` (LinearAlgebra.cpp) → result
vector/matrix created with the literal backing file `./_vals` / `./_vecs`
**[BUG F-024]** → CpuSolver::eig (LAPACK).

**`pc.save(A, "a.pycauset")`** → persistence.py writes header, payload and metadata
→ C++ copies or reuses the backing file (`copy_storage`).

---

## 6. Build, test and CI wiring
* `pyproject.toml` (scikit-build-core, pybind11 pinned to 2.12.0) → `CMakeLists.txt`
  (646 lines): finds or fetches OpenBLAS, LAPACKE, Eigen and OpenMP; builds
  `pycauset_core` (shared) and `_pycauset` (module); optional CUDA; `BUILD_TESTS`
  adds 6 gtest executables.
* Tests: `tests/python/` (130 files, pytest). Each file inserts `python/` into
  `sys.path`, so CI **copies the compiled .so into the source tree** before running.
  C++ tests: 9 files, but only 6 are wired into CMake, and **CI never builds or
  runs any C++ test**.
* CI: `ci.yml` (3 OS, py3.12, pytest + parity benchmark report), `asan.yml`
  (Linux ASan, main only), `docs.yml` (mkdocs gh-deploy), `publish.yml` (wheels).
* Docs: MkDocs Material (`mkdocs.yml`) over `documentation/`.
