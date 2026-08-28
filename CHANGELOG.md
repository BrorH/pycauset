# Changelog

All notable changes to PyCauset are documented in this file. The project follows
[Semantic Versioning](https://semver.org/). The documentation's **"Release 1 (R1)"**
milestone corresponds to **v0.5.1**.

## [Unreleased]: R1 (target v0.5.1)

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
