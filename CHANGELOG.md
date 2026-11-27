# Changelog

All notable changes to PyCauset are documented in this file. The project follows
[Semantic Versioning](https://semver.org/). The documentation's **"Release 1 (R1)"**
milestone corresponds to **v0.6.1**.

## [Unreleased]

Pre-alpha: these changes are breaking and come without compatibility shims.

### Changed
- Element-wise `+ - *` of upper triangular matrices return a `TriangularMatrix` (with a
  diagonal exactly when the result can be non-zero there), also with a diagonal or identity
  operand; these were dense results before. Values and dtypes unchanged.
- `matrix (+ - * /) vector` broadcasts the vector as a row, as NumPy does with a 1-D operand
  (it raised `TypeError`). `ndarray @ matrix` works (1-D left operand: a 1-D result).
- Indexing: `m[...]`, Python lists (`m[[0, 2], :]`) and NumPy integer scalars index as in NumPy.
- `&`, `|`, `^`, `~` work for bool and integer matrices and vectors of every structure and
  view, with NumPy's dtype rules (`pc.bitwise_and` on integers is an integer operation; it
  used to cast to bool). Float/complex operands raise `TypeError`.
- `TriangularMatrix(n, dtype="bool", has_diagonal=True)`: bit triangles with a diagonal.
  `UnitVector(n, k, dtype=...)` for every dtype.
- `invert` / `inverse` run in the engine for every dtype, structure and view (NumPy's
  algorithm, no NumPy fallback); the inverse of an upper triangular matrix is a
  `TriangularMatrix`; singular input raises `RuntimeError`.
- `eig`, `eigvals`, `eigh`, `eigvalsh`, `cholesky`, `qr` and `pinv` run in the engine for
  every dtype, structure and view, in NumPy's LAPACK dtype (single precision for
  float16/float32/complex32/complex64 input); no NumPy fallback. `eig`/`eigvals` always return
  complex results (NumPy returns a real array when every eigenvalue happens to be real).
  `eigh`/`eigvalsh`/`cholesky` read the lower triangle, as NumPy; `eigvalsh` no longer
  consults the `is_hermitian` property. `cholesky` of a matrix that is not positive definite
  raises `ValueError`. `qr` supports `mode='reduced'|'complete'|'r'`; `'raw'` raises.
- `lu`, `outer`, `cross`, `vecdot`, `norm(x, ord)`, `matrix_rank` and `cond` run in the
  engine for every dtype (`lu` was float32/float64 only; the others were NumPy). `outer`
  takes vectors of any lengths and follows NumPy's multiply dtype (it used to require equal
  lengths); `cross` follows NumPy's dtype (it was float64) and rejects bool.
- Fixed: `matrix_rank` of a triangular (e.g. causal) matrix counted its non-zero diagonal;
  a strictly triangular matrix of ones now has rank n - 1, not 0.
- Properties no longer change results anywhere: `eigvalsh` ignores `is_hermitian`, `solve`
  and `matrix_power` ignore `is_zero`, `norm` ignores `is_zero` / identity.
- `obj.properties` is removed (the boolean structure flags `is_symmetric`,
  `is_upper_triangular`, `is_zero`, ..., their validation and propagation, and the
  `properties` entry of saved files). Structure is the class (`TriangularMatrix`,
  `SymmetricMatrix`, `HermitianMatrix`, `DiagonalMatrix`, ...).
- `pc.symmetric`, `pc.antisymmetric`, `pc.diagonal` return `SymmetricMatrix`,
  `AntiSymmetricMatrix`, `DiagonalMatrix` in the input's exact dtype (complex too); new
  `pc.hermitian`. `solve_triangular` takes a `TriangularMatrix` (or its transpose),
  `DiagonalMatrix` or `IdentityMatrix`.
- Writing to a loaded matrix no longer changes the saved file (copy-on-write mapping, as the
  container contract states); `pc.save` replaces files atomically.
- `M @ v` / `v @ M` are matmul with a 1-D operand: integer results that overflow their dtype
  raise `OverflowError`, as integer matmul does (they used to wrap).
- Fixed: scaled views of transposed non-square matrices (`B.T * 2`) and transposed windows
  (`A[1:4, 2:6].T`) returned wrong values; `I @ v` for a float identity returned the first
  element only.
- `np.asarray(M)` exports every class in its own dtype (structured classes used to export as
  float64).
- File-backed operands (or ones above the IO streaming threshold) no longer make `matmul`
  and `invert` run element by element in Python through NumPy (matmul returned float for
  every dtype on that path).
- New structure x dtype combinations: `DiagonalMatrix` and `IdentityMatrix` in float16,
  float32, complex_float32/64; `TriangularMatrix`, `SymmetricMatrix`, `AntiSymmetricMatrix` in
  complex_float32/64; every structured class also in complex_float16 (interleaved float16
  pairs). New classes `HermitianMatrix` (`A == A.conj().T`) and
  `SkewHermitianMatrix` (`A == -A.conj().T`) for complex dtypes; a complex `SymmetricMatrix`
  is `A == A.T`. A combination that does not exist raises `ValueError` saying why.
- Complex division follows NumPy's algorithm (division by a signed zero gives NumPy's
  infinities). `A.conjugate()` is an alias of `A.conj()`.
- Results are never cached. `sum`, `trace`, `determinant`, `norm`, eigenvalues and inverses
  are computed on every call (cached values went stale after `M[i, j] = v`). Properties are
  metadata only: they never change a result (`is_zero=True` no longer makes `det` 0). Save
  files no longer have a `cached` section or `.objects/` sidecar.
- `~A` follows NumPy for structured matrices (every element is complemented, so the result
  is dense). `AntiSymmetricMatrix` requires a signed dtype.
- `solve`, `solve_triangular`, `lstsq`, `pinv`, `svd`, `svdvals` and `slogdet` run in the
  engine for every dtype, structure and view (no NumPy fallback); `svd` returns NumPy's
  shapes, `full_matrices=True` by default.
- Matrix products keep structure where the product has it: upper triangular @ upper
  triangular is a `TriangularMatrix` and diagonal @ diagonal a `DiagonalMatrix` for every
  real dtype with that class (was float64/int32 only); complex products use BLAS.
- `A += B` / `A += s` mean `A = A + B` (one code path; the separate in-place bindings are
  removed). float16 / complex-float16 matrices support `+ s` and `- s` with any real scalar.
- One class per structure; the element type is `A.dtype`: `DenseMatrix`, `TriangularMatrix`
  (causal matrices are dtype `bool`), `DiagonalMatrix`, `IdentityMatrix`, `SymmetricMatrix`,
  `AntiSymmetricMatrix`, `DenseVector`, `UnitVector`. The per-dtype classes (`FloatMatrix`,
  `IntegerMatrix`, `TriangularBitMatrix`, `Int16Vector`, ...) are removed.
- `zeros`, `ones`, `empty` without `dtype` return float64 (as NumPy); the dtype-deferred
  wrapper is removed.
- `pycauset.matrix(n)` no longer switches to float32 for n >= 10,000; nested lists are always
  dense (no structure inferred from values).
- Saved files record `structure` and `dtype`.
- Lazy results: integer `k * A` / `-A` are O(1) views; `C + C.T` / `C - C.T` of a causal matrix
  are read from its bits. Floating-point results of combined lazy steps may differ from
  step-by-step evaluation in the last digits (deterministic). `A.is_lazy`, `pycauset.materialize`.

- Vector arithmetic follows the matrix (NumPy) dtype rules: `int8` vector `+ 1` stays `int8`
  (was float64); `v / 2`, `2 - v`, `-v`, `v * w`, `v / w` are supported; integer `dot`
  returns an exact `int`. `2 / A` works for matrices.
- `import pycauset` sets `OMP_WAIT_POLICY=PASSIVE` and `OPENBLAS_THREAD_TIMEOUT=4` unless
  already set (thread pools no longer starve each other; see the Performance Guide).
- Warnings are deduplicated by Python's warnings filter only (no hidden "warn once" state);
  the int32 accumulator notices are removed.

### Removed
- `MatrixBase.multiply(other)`: an undocumented alias of `@`; use `a @ b` or `a.matmul(b)`.
- `pc.set_io_streaming_threshold` / `pc.get_io_streaming_threshold`: they changed only what the
  IO trace *said*; whether an operation runs out of core is the engine's decision (F-100).
  `pc.last_io_trace()` now reports facts: operand shape/dtype/storage, the kernels the engine
  ran, and `streamed`. The keys `route`, `reason`, `tile_shape`, `queue_depth`, `events` and
  `storage` are gone with the Python planner that produced them.
- `precision_mode`, the lazy expression engine (`LazyMatrix`), `pycauset.cuda` and the `[gpu]`
  extra (CUDA moved to `experimental/cuda`, on hold), the per-class buffer protocol.

### Fixed
- `pycauset.sum` returns `int` for integer/bool inputs (exact), `float` for real, `complex` for
  complex (was always `complex`).
- Many silent wrong results (review/FINDINGS.md F-001 ... F-058), among them: integer scalar
  multiplication, `C + C.T` dropping the lower triangle, views ignored by `.get`, writes and
  several kernels, int64 precision in matvec and element-wise ops, false links in
  `CausalSet.links()`.

## [0.6.1] - 2026-08-28

### Added
- `pinv`: Moore-Penrose pseudoinverse (normal-equations baseline + NumPy SVD fallback).
- `load_matrix(path)`: alias of `load`.
- `svdvals`, `matrix_rank`, `matrix_power`, `outer`: NumPy-equivalent linalg
  operations, each with structural shortcuts where a closed form exists (rank of
  identity/diagonal/triangular, power of identity/zero/diagonal, norm of identity/zero).
- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_nand`, `bitwise_nor`,
  `bitwise_xnor`: elementwise logic operations on bit matrices/vectors, always returning
  a bit structure (DenseBitMatrix/BitVector).
- `cross`: 3D cross product.
- `vecdot`: conjugate dot product (complex-aware inner product).
- `symmetric(data)` and `antisymmetric(data)`: validated constructors for symmetric
  (`A == A.T`) and anti-symmetric (`A == -A.T`) matrices. Float input produces a native
  `SymmetricMatrix`/`AntiSymmetricMatrix` with packed upper-triangle storage (roughly 2x
  smaller than dense); integer/bool input produces a dense matrix with
  `is_symmetric`/`is_anti_symmetric` asserted (exact storage, no packing in R1).
- `diagonal(data)`: constructs a `DiagonalMatrix` from a 1D vector of entries or a 2D
  square matrix (extracting its diagonal).
- `det` and `rank`: NumPy-compatible aliases for `determinant` (`np.linalg.det`) and
  `matrix_rank`.
- Complete per-operation support-status registry (`OpRegistry`) now covering the vector
  and vector-scalar ops (`dot`, `add_vector`, `subtract_vector`, `outer`, `add_scalar`,
  `mul_scalar`) in addition to the matrix ops.
- `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md` (Eigen MPL2, OpenBLAS BSD, pybind11 BSD,
  googletest BSD, build-time deps, CUDA-not-bundled).
- GitHub Actions CI matrix (`ci.yml`): Windows / macOS / Linux on Python 3.12.
- Benchmark harness (`benchmarks/bench.py`, `benchmarks/plot.py`) with graphs, and the
  out-of-core RAM-limit demo (`benchmarks/bench_ram.py`).
- `bool`: alias for the `bool_` dtype token (both normalize to `"bool"`). `pc.bool`
  now resolves and is star-exported; `bool_` remains for back-compat.
- `zeros`/`ones`/`empty` no longer require `dtype=`. With no dtype they return a
  dtype-deferred wrapper that resolves to `int32` on first use (or to the dtype of
  the first written value); a still-typeless `empty` raises on read/use.
- `dot(a, b)` now accepts matrices and matches `np.dot` semantics (vector-vector
  scalar, matrix-matrix matmul, matrix-vector / vector-matrix products).
- "constant" structure registered in the properties/structure system
  (`is_constant` + `constant_value`); `ones` without a dtype carries it.
- Everyday-use scenario tests (`tests/python/test_everyday_scenarios.py`), each
  exercised in at least two ways and cross-checked against NumPy, plus a findings
  note (`documentation/dev/FINDINGS_EVERYDAY_SCENARIOS.md`) and a scenario
  benchmark script (`benchmarks/scenario_benchmarks.py`).

### Fixed
- Fixed Linux memory detection: `MemoryGovernor` used `sysinfo().freeram`
  (MemFree), which excludes reclaimable page cache and reads artificially low
  right after a build, so `can_fit_in_ram()`/`request_ram()` wrongly routed small
  and bit-matrix operations to disk/streaming (`test_ram_backed_small_object`,
  `test_scalar_optimization`, `test_scale_first_bit_guard` failures on Linux). It
  now reads `MemAvailable` from `/proc/meminfo`, with a `sysinfo()` fallback.
- Fixed a use-after-free in the memory governor: `PersistentObject`'s destructor
  never unregistered the object from `MemoryGovernor`, so its LRU accumulated
  dangling `PersistentObject*` entries and `evict_until_fits()` dereferenced them via
  `spill_to_disk()` under memory pressure. This caused intermittent constructor
  crashes (Linux SIGSEGV, macOS SIGBUS) during the large-allocation stress test.
  `~PersistentObject()` now unregisters.
- Bool/bit matmul no longer raises `Unsupported matrix multiplication types`:
  `DenseBitMatrix @ TriangularBitMatrix` and `TriangularBitMatrix @ DenseBitMatrix`
  now promote to a dense `int32` result (bit x bit counting semantics).
- Cubing a causal matrix now works: `TriangularIntegerMatrix x TriangularBitMatrix`
  and `TriangularIntegerMatrix x TriangularIntegerMatrix` matmul are supported
  (native `int32` result), so `(C @ C) @ C` and `matrix_power(C, 3)` no longer throw
  `Unsupported matrix multiplication types`.
- `TriangularIntegerMatrix` (the result of `C @ C`) now exports to NumPy as
  `int32` instead of raising `TypeError: data type 'triangularinteger' not
  understood`.
- Dense mixed-dtype matmul now works via a general promotion fallback:
  `int32 @ float64`, `float64 @ int32`, `int32 @ TriangularBitMatrix`, and
  `float64 @ TriangularBitMatrix` (and their reverses) no longer raise
  `Unsupported matrix multiplication types`.
- `matrix_power(C, k)` keeps triangular causal-matrix powers triangular (it no
  longer densifies to a dense `IntegerMatrix` for some powers).
- `dot(vector, matrix)` returns a 1-D vector like `np.dot` (previously a
  transposed `(1, n)` row vector).
- Wide mixed-integer matmul now promotes to the wider type instead of raising:
  `int8 @ int16`, `int64 @ int32`, `uint8 @ uint16`, `uint32 @ int32` (promotes to
  `int64`), and other mixed-width integer combinations all work via a generic
  integer fallback kernel.
- `norm(matrix, ord=2)` now returns the spectral norm (largest singular value); it
  previously returned the Frobenius norm for `ord=2`.
- `np.asarray(IdentityMatrix)` no longer raises `TypeError: data type 'identity' not
  understood`; identity matrices export as float64.
- Structural shortcuts now trigger for native structural types: `matrix_rank` of an
  `IdentityMatrix`/`DiagonalMatrix` uses a closed form instead of an SVD.
- `SymmetricMatrix`/`AntiSymmetricMatrix` (present in the native core but previously
  unreachable from Python) are now integrated end to end: recognized by the structure
  system (no longer "general"), exported to NumPy without `TypeError: data type
  'symmetric' not understood`, routed through a dense float64 fallback in `matmul`
  (no longer `Unsupported matrix multiplication types`), and round-tripped by
  `save()`/`load()` (no longer `Unknown matrix type: None`).
- Documentation/export completeness: `dot`, `load`, `load_matrix`, `to_numpy`,
  `set_export_max_bytes`, `CausalSet`, and `TriangularMatrix` were documented and
  callable but missing from `__all__`, so `from pycauset import *` dropped them; they
  are now exported. Reference pages were added for `cross`, `determinant`,
  `matrix_power`, `matrix_rank`, `outer`, `qr`, `svdvals`, `trace`, `vecdot`, the six
  `bitwise_*` ops, and `DiagonalMatrix`.
- `CausalSpacetime` is documented at its real location (`pycauset.CausalSpacetime`, a
  native abstract type with no R1 constructor/methods), not the non-existent
  `pycauset.spacetime.CausalSpacetime`.
- Removed a leftover `std::cerr` debug print in `MemoryMapper` that spammed
  "mmap success: ..." on every memory mapping.
- Fixed a macOS bug where `:memory:` matrices had no file descriptor (macOS has no
  `memfd_create`), so `TriangularBitMatrix` random generation crashed with
  "File descriptor invalid" in `map_region()`. macOS now uses POSIX `shm_open` to give
  anonymous memory a real fd, matching Linux's `memfd` behavior.
- Dense `float64` `solve` and `lu` now route through LAPACK (`dgesv`/`dgetrf`) instead of
  a naive scalar Gaussian elimination; this also made the float64 path consistent with the
  float32 path and with NumPy's singularity behavior.
- `matrix([[1+2j, …]])` (complex *list* input) now constructs a real complex matrix
  (`ComplexFloat64Matrix`) instead of a broken abstract `Matrix` with no dtype.
- `matrix(ndarray, backing_file=…)` no longer silently returns an in-memory matrix; it
  raises a clear `TypeError` directing users to `save()`/`load()`.
- SIMD kernels are baseline-safe: `-march=native` removed and per-function
  `__attribute__((target(...)))` added, so built binaries no longer crash with an illegal
  instruction on CPUs without AVX2/AVX-512.
- Eigen decomposition correctness across a `save()`/`load()` round-trip is now pinned.

### Changed
- Integer overflow policy is now documented and explicit: elementwise integer arithmetic
  wraps silently (C/NumPy two's-complement semantics), while `matmul` reductions raise
  `OverflowError`. (Previously the docs promised "hard error everywhere" but only `matmul`
  actually enforced it.)
- `__all__` no longer star-exports internal native machinery (`LazyMatrix`, `lazy_*`,
  `MemoryGovernor`, `IOAccelerator`, `OpContract`, `OpRegistry`, `get_storage_root`,
  `set_storage_root`, `make_coordinates`, `sprinkle`). These stay reachable as
  `pycauset.<name>` for advanced use, but are no longer presented as public API.

### Performance
- `DenseBitMatrix x TriangularBitMatrix` matmul now uses a popcount kernel instead
  of a naive element loop (~15x faster at n=256).
- `C @ vector` matvec for `TriangularBitMatrix` now uses scale-first kernels that
  iterate the set bits (beats NumPy's dense int64 matvec by ~30x at N=5000).
- R1 goal restated to allow targeted optimization of correctness-first paths where a
  naive fallback is egregiously slow relative to NumPy.
