# Changelog

All notable changes to PyCauset are documented in this file. The project follows
[Semantic Versioning](https://semver.org/). The documentation's **"Release 1 (R1)"**
milestone corresponds to **v0.5.1**.

## [Unreleased] — R1 (target v0.5.1)

### Added
- `pinv` — Moore–Penrose pseudoinverse (normal-equations baseline + NumPy SVD fallback).
- `load_matrix(path)` — alias of `load`.
- Complete per-operation support-status registry (`OpRegistry`) now covering the vector
  and vector-scalar ops (`dot`, `add_vector`, `subtract_vector`, `outer`, `add_scalar`,
  `mul_scalar`) in addition to the matrix ops.
- `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md` (Eigen MPL2, OpenBLAS BSD, pybind11 BSD,
  googletest BSD, build-time deps, CUDA-not-bundled).
- GitHub Actions CI matrix (`ci.yml`): Windows / macOS / Linux on Python 3.12.

### Fixed
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
