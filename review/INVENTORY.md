# Inventory

Everything in the repository, what it's for, and its review status. This is the
checklist for PLAN Phase 3 ("look at every line"): tick a file (☐ → ☑) only when it
has been reviewed against the PLAN checklist **and** its notes are in
`review/areas/<area>.md`. Use ⚠ for "reviewed, has open findings".

Regenerate the source tables with `python review/scripts/inventory_table.py` after
moves, keeping the Review/Notes columns. Snapshot: commit `046c5ab`, 2026-11-24.

## Size at a glance

| Area | Files | Lines |
|---|---:|---:|
| C++ sources `src/` (incl. CUDA 4.3k) | 56 | 25,025 |
| C++ headers `include/` | 59 | 6,077 |
| Python package `python/` | 38 | 14,336 |
| Tests `tests/` (130 py, 9 cpp) | 145 | 16,708 |
| Documentation `documentation/` (md) | 263 | 21,336 |
| Benchmarks / tools / demos / scripts | 28 | 2,164 |

## Top-level layout

| Path | What it is | Status / notes |
|---|---|---|
| `include/pycauset/` | C++ public headers (core, matrix, vector, compute, math, causet) | live |
| `src/` | C++ implementation + pybind11 bindings + CUDA plugin | live; CUDA on hold |
| `python/pycauset/` | Python package (facade, physics suite, `_internal/` glue) | live |
| `tests/python/` | pytest suite, 130 files (incl. 2 non-tests) | 799 pass / 34 skip |
| `tests/*.cpp` | gtest unit tests (9; 6 wired into CMake) | F-010 |
| `tests/manual/`, `tests/stability/` | ad-hoc scripts, not run anywhere | delete or convert |
| `tests/BUG_LOG.md` | historical bug log (6 entries, Aug 2026) | merge with FINDINGS |
| `benchmarks/` | 11 benchmark scripts; `r2_parity.py` is the CI one | consolidate |
| `demos/` | 7 short conference demos + README | live, not run in CI |
| `tools/` | audits (`audit_public_surface`, `check_native_exports`, `check_support_matrix`), PowerShell build/release/CUDA install, one-off diagnostics | mixed |
| `scripts/make_r1_gallery.py` | regenerates README gallery images | move to tools |
| `mockups/` | one HTML mockup ("Spacetime Studio") | delete |
| `documentation/` | MkDocs site source (see below) | ~20 broken snippets (F-008) |
| `overrides/` | MkDocs theme overrides | live |
| `mkdocs.yml`, `requirements-docs.txt` | docs build | live |
| `CMakeLists.txt` | C++ build (646 lines) | dead targets, debug spam |
| `pyproject.toml` | scikit-build-core packaging, ruff/mypy config | py3.8 floor is fiction |
| `.github/workflows/` | `ci.yml`, `asan.yml`, `docs.yml`, `publish.yml` | no C++ tests in CI |
| `README.md`, `CHANGELOG.md`, `BENCHMARKS.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE` | root docs | claims to re-verify (Phase 5) |
| `review/` | **this overhaul's workspace** | — |

## Documentation tree

| Path | Contents | Notes |
|---|---|---|
| `documentation/guides/` | 20 user guides | snippets mostly run; see F-008 |
| `documentation/docs/` | API reference: 96 function pages, class pages, parameters, assets | 3 pages for non-existent symbols; many signature blocks fenced as python |
| `documentation/internals/` | design docs (algorithms, compute, dtype, memory, streaming, lazy) | verify against ARCHITECTURE.md |
| `documentation/internals/plans/` (+ `archive/`, 18 files) | R1-era plans, TODO, optimization status | history; proposed for deletion |
| `documentation/dev/` | 16 developer notes (build, bindings, container format, QA audits, perf findings) | overlaps internals |
| `documentation/project/` | philosophy, public API contract, protocols (8), R2 plans (7) | protocols worth keeping |

## Source files

### C++ headers: core

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `include/pycauset/core/DebugTrace.hpp` | 18 | ☐ |thread_local kernel-trace hook for tests |
| `include/pycauset/core/Float16.hpp` | 168 | ☐ |software float16_t |
| `include/pycauset/core/IOAccelerator.hpp` | 65 | ☐ |prefetch/madvise from MemoryHints |
| `include/pycauset/core/MatrixTraits.hpp` | 96 | ☐ |type traits per matrix class |
| `include/pycauset/core/MatrixTypeResolver.hpp` | 21 | ☐ |result structure for binary ops |
| `include/pycauset/core/MemoryGovernor.hpp` | 219 | ☐ |global RAM budget + LRU spill (leaked singleton) |
| `include/pycauset/core/MemoryHints.hpp` | 59 | ☐ |Lookahead Protocol hint types |
| `include/pycauset/core/MemoryMapper.hpp` | 58 | ☐ |RAM buffer or mmap; pinning; offset maps into containers |
| `include/pycauset/core/Nvtx.hpp` | 43 | ☐ |NVTX profiling stubs (CUDA) |
| `include/pycauset/core/ObjectFactory.hpp` | 90 | ☐ |create/load matrices & vectors by (shape, DataType, MatrixType) |
| `include/pycauset/core/OpRegistry.hpp` | 34 | ☐ |per-op support contracts (introspection, not dispatch) |
| `include/pycauset/core/ParallelUtils.hpp` | 159 | ☐ |ThreadPool + ParallelFor/BlockMap. F-004 deadlock, F-005 |
| `include/pycauset/core/PersistentObject.hpp` | 171 | ☐ |root of all storage objects; view metadata (scalar/transposed/conjugated); CoW. Global namespace |
| `include/pycauset/core/PromotionResolver.hpp` | 57 | ☐ |dtype promotion rules + underpromotion warning |
| `include/pycauset/core/PropertyFlags.hpp` | 35 | ☐ |bit flags mirroring Python properties (2 sources of truth) |
| `include/pycauset/core/ScalarUtils.hpp` | 53 | ☐ |scalar helpers |
| `include/pycauset/core/StorageUtils.hpp` | 22 | ☐ |global storage root + memory threshold |
| `include/pycauset/core/StructureResolver.hpp` | 57 | ☐ |structure algebra (diag∘X→diag ...) |
| `include/pycauset/core/SystemUtils.hpp` | 32 | ☐ |available RAM detection |
| `include/pycauset/core/Types.hpp` | 42 | ☐ |DataType + MatrixType enums (MatrixType mixes structure & dtype) |

### C++ headers: matrix / vector / expression

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `include/pycauset/matrix/ComplexFloat16Matrix.hpp` | 231 | ☐ |hand-written complex16 (two planes) |
| `include/pycauset/matrix/DenseBitMatrix.hpp` | 106 | ☐ | |
| `include/pycauset/matrix/DenseMatrix.hpp` | 561 | ☐ |DenseMatrix<T> + bit specialization; multiply_scalar view (F-001) |
| `include/pycauset/matrix/DiagonalMatrix.hpp` | 218 | ☐ |O(n) diagonal |
| `include/pycauset/matrix/IdentityMatrix.hpp` | 186 | ☐ |O(1) identity |
| `include/pycauset/matrix/MatrixBase.hpp` | 193 | ☐ |abstract matrix interface |
| `include/pycauset/matrix/MatrixOps.hpp` | 194 | ☐ |free-function op helpers |
| `include/pycauset/matrix/SymmetricMatrix.hpp` | 248 | ☐ |packed sym/antisym |
| `include/pycauset/matrix/TriangularBitMatrix.hpp` | 92 | ☐ |alias header for causal matrix |
| `include/pycauset/matrix/TriangularMatrix.hpp` | 280 | ☐ |TriangularMatrix<T>, base; <bool> = causal matrix |
| `include/pycauset/matrix/expression/BinaryExpression.hpp` | 85 | ☐ | |
| `include/pycauset/matrix/expression/Functors.hpp` | 75 | ☐ | |
| `include/pycauset/matrix/expression/MatrixExpression.hpp` | 116 | ☐ | |
| `include/pycauset/matrix/expression/MatrixExpressionWrapper.hpp` | 145 | ☐ | |
| `include/pycauset/matrix/expression/MatrixRefExpression.hpp` | 48 | ☐ | |
| `include/pycauset/matrix/expression/ScalarExpression.hpp` | 46 | ☐ | |
| `include/pycauset/matrix/expression/UnaryExpression.hpp` | 72 | ☐ | |
| `include/pycauset/vector/ComplexFloat16Vector.hpp` | 171 | ☐ |complex16 vector |
| `include/pycauset/vector/DenseVector.hpp` | 186 | ☐ |DenseVector<T> + BitVector |
| `include/pycauset/vector/UnitVector.hpp` | 116 | ☐ |one-hot vector |
| `include/pycauset/vector/VectorBase.hpp` | 63 | ☐ | |

### C++ headers: compute / math / causet

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `include/pycauset/compute/AcceleratorConfig.hpp` | 14 | ☐ |accelerator config |
| `include/pycauset/compute/AutoSolver.hpp` | 134 | ☐ |CPU/GPU router |
| `include/pycauset/compute/ComputeContext.hpp` | 57 | ☐ |singleton owning AutoSolver |
| `include/pycauset/compute/ComputeDevice.hpp` | 140 | ☐ |fat device interface (~40 virtuals) |
| `include/pycauset/compute/ComputeWorker.hpp` | 65 | ☐ |tile-execution interface |
| `include/pycauset/compute/ComputeWorkerExtensionTemplate.hpp` | 35 | ☐ |how-to-add-op template header |
| `include/pycauset/compute/HardwareProfile.hpp` | 36 | ☐ |device benchmark profile |
| `include/pycauset/compute/MatrixPropertyTraits.hpp` | 61 | ☐ |property traits for kernels |
| `include/pycauset/compute/StreamingManager.hpp` | 56 | ☐ |C++ tiled out-of-core driver (vs Python one) |
| `include/pycauset/compute/cpu/CpuComputeWorker.hpp` | 38 | ☐ |CPU tile worker |
| `include/pycauset/compute/cpu/CpuDevice.hpp` | 76 | ☐ |CPU device adapter |
| `include/pycauset/compute/cpu/CpuSolver.hpp` | 86 | ☐ |CPU kernel class decl |
| `include/pycauset/math/Eigen.hpp` | 13 | ☐ |eigen helpers |
| `include/pycauset/math/LinearAlgebra.hpp` | 68 | ☐ |pycauset::add/matmul/... entry points |
| `include/pycauset/causet/Spacetime.hpp` | 169 | ☐ |native Minkowski spacetimes; stray #pragma message (F-019) |
| `include/pycauset/causet/Sprinkler.hpp` | 41 | ☐ |native sprinkling |

### C++ sources

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `src/bindings/bind_causet.cpp` | 110 | ☐ |native spacetimes, sprinkle |
| `src/bindings/bind_complex.cpp` | 174 | ☐ | |
| `src/bindings/bind_core.cpp` | 273 | ☐ |storage/threads/IO config |
| `src/bindings/bind_cuda.cpp` | 53 | ☐ |_pycauset_cuda module (on hold) |
| `src/bindings/bind_expression.cpp` | 260 | ☐ |LazyMatrix |
| `src/bindings/bind_math.cpp` | 52 | ☐ | |
| `src/bindings/bind_matrix.cpp` | 4719 | ☐ |ALL matrix classes + operators + NumPy buffer. Split candidate |
| `src/bindings/bind_op_registry.cpp` | 23 | ☐ | |
| `src/bindings/bind_vector.cpp` | 1371 | ☐ |vector classes |
| `src/bindings/binding_warnings.cpp` | 78 | ☐ | |
| `src/bindings.cpp` | 25 | ☐ |PYBIND11_MODULE entry |
| `src/causet/Sprinkler.cpp` | 118 | ☐ |sprinkling + interrupt |
| `src/compute/AutoSolver.cpp` | 1100 | ☐ |router impl |
| `src/compute/ComputeContext.cpp` | 170 | ☐ | |
| `src/compute/StreamingManager.cpp` | 333 | ☐ | |
| `src/compute/cpu/CpuComputeWorker.cpp` | 45 | ☐ | |
| `src/compute/cpu/CpuDevice.cpp` | 229 | ☐ | |
| `src/compute/cpu/CpuSolver.cpp` | 5867 | ☐ |ALL CPU kernels, monolith (F-006 UB at :165). Split candidate |
| `src/core/DebugTrace.cpp` | 32 | ☐ | |
| `src/core/IOAccelerator.cpp` | 224 | ☐ | |
| `src/core/MemoryGovernor.cpp` | 354 | ☐ | |
| `src/core/MemoryMapper.cpp` | 704 | ☐ |_GNU_SOURCE redefined (F-019) |
| `src/core/ObjectFactory.cpp` | 586 | ☐ | |
| `src/core/OpRegistration.cpp` | 202 | ☐ |OpRegistry contents |
| `src/core/ParallelUtils.cpp` | 64 | ☐ | |
| `src/core/PersistentObject.cpp` | 430 | ☐ | |
| `src/core/PromotionResolver.cpp` | 297 | ☐ | |
| `src/core/StorageUtils.cpp` | 48 | ☐ | |
| `src/core/SystemUtils.cpp` | 91 | ☐ | |
| `src/math/Eigen.cpp` | 17 | ☐ | |
| `src/math/LinearAlgebra.cpp` | 1142 | ☐ |op entry: allocate result, dispatch |
| `src/matrix/DenseBitMatrix.cpp` | 281 | ☐ | |
| `src/matrix/MatrixBase.cpp` | 111 | ☐ | |
| `src/matrix/MatrixOps.cpp` | 10 | ☐ | |
| `src/matrix/TriangularBitMatrix.cpp` | 482 | ☐ | |
| `src/matrix/TriangularMatrix.cpp` | 44 | ☐ | |
| `src/vector/DenseVector.cpp` | 123 | ☐ | |
| `src/vector/VectorBase.cpp` | 52 | ☐ | |

### C++ binding helpers

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `src/bindings/binding_warnings.hpp` | 24 | ☐ | |
| `src/bindings/bindings_common.hpp` | 15 | ☐ | |
| `src/bindings/numpy_export_guard.hpp` | 153 | ☐ | |

### CUDA (on hold)

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `src/accelerators/cuda/ArnoldiDriver.cu` | 195 | ☐ | |
| `src/accelerators/cuda/ArnoldiDriver.hpp` | 20 | ☐ | |
| `src/accelerators/cuda/AsyncStreamer.hpp` | 153 | ☐ | |
| `src/accelerators/cuda/CholeskyDriver.cu` | 141 | ☐ | |
| `src/accelerators/cuda/CholeskyDriver.hpp` | 19 | ☐ | |
| `src/accelerators/cuda/CudaDevice.cu` | 2102 | ☐ | |
| `src/accelerators/cuda/CudaDevice.hpp` | 172 | ☐ | |
| `src/accelerators/cuda/CudaFuture.hpp` | 49 | ☐ | |
| `src/accelerators/cuda/CudaSolver.cu` | 1429 | ☐ | |
| `src/accelerators/cuda/CudaSolver.hpp` | 82 | ☐ | |
| `src/accelerators/cuda/MatmulDriver.cu` | 27 | ☐ | |
| `src/accelerators/cuda/MatmulDriver.hpp` | 30 | ☐ | |
| `src/accelerators/cuda/ScopedPinner.hpp` | 44 | ☐ | |

### Python package

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `python/pycauset/__init__.py` | 2416 | ☐ |public facade + factories + monkeypatching. 2.4k lines, target <300 |
| `python/pycauset/_storage.py` | 130 | ☐ |temp-file cleanup; StorageRegistry dead |
| `python/pycauset/_version.py` | 24 | ☐ |generated by setuptools_scm at build time (gitignored) |
| `python/pycauset/causet.py` | 486 | ☐ |CausalSet |
| `python/pycauset/cuda.py` | 52 | ☐ |CUDA facade (on hold) |
| `python/pycauset/field.py` | 335 | ☐ |fields, propagators, SJ state |
| `python/pycauset/spacetime.py` | 1019 | ☐ |Spacetime ABC, registry, library, curved |
| `python/pycauset/synthetic.py` | 95 | ☐ |poset generators |
| `python/pycauset/vis.py` | 435 | ☐ |Plotly plots |
| `python/pycauset/__init__.pyi` | 342 | ☐ |STALE stub, 71 names missing (F-007) |

### Python internals

| File | Lines | Review | Notes |
|---|---:|:-:|---|
| `python/pycauset/_internal/__init__.py` | 4 | ☐ | |
| `python/pycauset/_internal/big_blob_cache.py` | 159 | ☐ | |
| `python/pycauset/_internal/blockmatrix.py` | 852 | ☐ |Python BlockMatrix |
| `python/pycauset/_internal/coercion.py` | 57 | ☐ | |
| `python/pycauset/_internal/dense_python.py` | 90 | ☐ | |
| `python/pycauset/_internal/dtypes.py` | 167 | ☐ | |
| `python/pycauset/_internal/export_guard.py` | 319 | ☐ |NumPy export size/disk guard (F-016) |
| `python/pycauset/_internal/factories.py` | 247 | ☐ | |
| `python/pycauset/_internal/formatting.py` | 104 | ☐ | |
| `python/pycauset/_internal/gpu_hint.py` | 66 | ☐ | |
| `python/pycauset/_internal/io_observability.py` | 335 | ☐ | |
| `python/pycauset/_internal/lazy_allocation.py` | 200 | ☐ | |
| `python/pycauset/_internal/linalg_cache.py` | 140 | ☐ | |
| `python/pycauset/_internal/matrix_api.py` | 618 | ☐ |TriangularMatrix front etc. |
| `python/pycauset/_internal/native.py` | 141 | ☐ | |
| `python/pycauset/_internal/native_setup.py` | 72 | ☐ |OpenBLAS threads via ctypes, GPU hint |
| `python/pycauset/_internal/native_threshold.py` | 41 | ☐ | |
| `python/pycauset/_internal/ops.py` | 1691 | ☐ |linalg front-ends; NumPy fallbacks (F-011); cond (F-003) |
| `python/pycauset/_internal/patching.py` | 102 | ☐ |operator wrappers (temp tracking) |
| `python/pycauset/_internal/persistence.py` | 1664 | ☐ |.pycauset container I/O |
| `python/pycauset/_internal/properties.py` | 1199 | ☐ |property system + operator wrappers |
| `python/pycauset/_internal/runtime.py` | 141 | ☐ |storage root (./.pycauset, F-017), temp tracking |
| `python/pycauset/_internal/storage_debug.py` | 97 | ☐ |debug tool; only own test uses it |
| `python/pycauset/_internal/streaming_manager.py` | 288 | ☐ |Python streaming planner (dup of C++ concept) |
| `python/pycauset/_internal/submatrix_view.py` | 155 | ☐ | |
| `python/pycauset/_internal/support_matrix.py` | 148 | ☐ |support table; tests/tools only |
| `python/pycauset/_internal/thunks.py` | 125 | ☐ | |
| `python/pycauset/_internal/warnings.py` | 27 | ☐ | |
## Tests (`tests/python/`), grouped by subject

Milestone-named files (`*_r2`, `*_phase_*`, `phase3_4_*`, `test_r1_*`) are listed by
subject so the Phase 4 reorganization has a starting map.

| Subject | Files |
|---|---|
| Construction / dtypes | `test_typed_numpy_constructors`, `test_float_types`, `test_complex_float_dtypes`, `test_complex_vector_dtypes`, `test_complex_scalar`, `test_dtype_deep_dive`, `test_types_extensive`, `test_factories_scalar_rejection`, `test_identity`, `test_unit_vector`, `test_symmetric`, `test_matrix_hierarchy`, `test_promotion_resolver` |
| Elementwise / operators | `test_operations`, `test_operations_extensive`, `test_elementwise_r2`, `test_bit_mixed_fastpaths`, `test_scalar_optimization`, `test_scale_first_bit_guard`, `test_sum`, `test_norm`, `test_conjugation`, `test_matrix_transpose`, `test_transposed_vectors`, `test_vector`, `test_vectors_advanced`, `test_inner_product_types` |
| Matmul | `test_matmul_transpose`, `test_hybrid_matmul` (GPU), `test_integer_matmul_overflow_*` (2), `test_float_underpromotion_warning`, `test_k_matrix` |
| Linear algebra | `test_inverse_{correctness,cpu,cpu_block,extensive}`, `test_factorizations`, `test_eigen_{cpu,caching,block_matrix,op_contract,streaming}`, `test_eig_general_cpu`, `test_arnoldi`, `test_skew`, `test_skew_comprehensive`, `test_linalg_phase_g_endpoints` |
| Lazy evaluation | `test_lazy_evaluation`, `test_lazy_ops_comprehensive` |
| Properties | `test_properties_*` (9 files) |
| Storage / out-of-core | `test_storage`, `test_storage_extensive`, `test_storage_kwarg`, `test_storage_hard_break`, `test_storage_crash_consistency`, `test_storage_debug_tool`, `test_ram_storage`, `test_backing_dir_api`, `test_temp_storage_lifecycle`, `test_outofcore_{elementwise,general,lu}`, `test_streaming_manager`, `test_io_consistency`, `test_io_observability`, `test_numpy_import_max_in_ram_bytes` |
| Persistence | `test_snapshot_phase1_roundtrip`, `test_snapshot_phase2_metadata_stability`, `test_convert_file`, `test_caching` |
| Block matrices | `test_blockmatrix_phase_{b,c_refine,c_views,d_elementwise,d_thunks,e_persistence,f_integration,h_hardening}` |
| NumPy interop | `test_numpy_{integration,interop,interop_ergonomics,conversion_policy,vector_complex_import}`, `test_interop_extensive` |
| Public API / contracts | `test_public_api`, `test_interface`, `test_registry_binding`, `test_phase7_op_contract`, `test_phase1_inventory`, `test_support_matrix`, `test_known_bugs` |
| Safety / edge cases | `test_r1_safety_comprehensive`, `test_edge_cases_core`, `test_everyday_scenarios`, `test_behaviour` |
| Physics | `test_spacetime`, `test_spacetime_extensive`, `test_spacetime_r2_abc`, `test_minkowski_r2`, `test_curved_r2`, `test_bh_r2`, `test_create_r2`, `test_coordinates`, `test_causet_r2_sprinkle_validate`, `test_struct_r2`, `test_synth_r2`, `test_dim_r2`, `test_batch_r2`, `test_field_r2`, `test_state_r2`, `test_ent_r2`, `test_conv_continuum_r2` |
| Visualization | `test_vis`, `test_viz_r2` |
| GPU (all skip here) | `test_gpu`, `test_gpu_factorizations`, `test_gpu_features`, `test_hybrid_correctness_suite`, `test_cuda_hardware_profile_mock` |
| Not tests | `debug_load.py` (CUDA DLL loader), `phase3_4_verification.py` (unittest script writing into cwd) |
