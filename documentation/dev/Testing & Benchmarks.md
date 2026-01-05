# Testing & Benchmarks

This page documents how to validate correctness and performance.

## Test layers

### 1) Python tests (primary user-surface validation)

Location:
- `tests/python/`

These tests validate:
- top-level `pycauset.*` API behavior,
- interoperability (NumPy integration),
- storage/persistence behavior,
- GPU feature gating,
- out-of-core behaviors.

### 2) C++ unit tests (engine invariants)

Location:
- `tests/*.cpp`

These tests validate:
- memory governor behavior,
- I/O accelerator behavior,
- core matrix invariants.

## Benchmarks

Location:
- `benchmarks/`

Benchmarks exist to compare:
- PyCauset vs NumPy for in-memory matrices,
- direct vs streaming paths,
- CPU vs GPU paths.

Quick smoke harness:
- `benchmarks/benchmark_io_smoke.py` measures save/load wall time and MB/s for a configurable square size (env `PYCAUSET_IO_BENCH_SIZE`). It is CI-friendly and useful for regression spot checks.

Recommended baseline:
- run the “CPU vs NumPy” benchmark suite after solver changes.

## Protocol

- Correctness first: tests must pass before trusting benchmarks.
- Benchmarks should be run with stable conditions:
  - consistent seeds,
  - clear dtype,
  - documented hardware.

## Link to optimization tracking

The authoritative checklist for dtype/op coverage and readiness gates is:

- `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md`

When a checklist item changes status:
- ensure a corresponding test exists (or is added),
- ensure a benchmark script exists (or is updated).

### 3) Safety Tests (R1_SAFETY)

Location:
- `tests/python/test_safety.py` (Basic smoke tests)
- `tests/python/test_r1_safety_comprehensive.py` (Extensive stress/fuzzing suite)

These tests validate:
- **Corrupt Load**: Ensuring `pc.load()` rejects files with invalid headers or magic bytes (Fuzzing 50+ iterations).
- **Spill Integrity**: Verifying that internal `.tmp` files (with Simple Headers) are read correctly.
- **Leak Detection**: Verifying that large alloc/free cycles do not cause OOM (validating `OfferVirtualMemory` logic).
- **Concurrency**: Threaded I/O stress testing to ensure thread-safety of file operations.
- **Persistence**: Verifying physical disk writes via explicit flush checks.
