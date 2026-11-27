# Testing & Benchmarks

This page documents how to validate correctness and performance.

## Test layers

Run everything with `PYTHONPATH=python python -m pytest -q -p no:cacheprovider` (the
`testpaths` in `pyproject.toml`), plus `PYCAUSET_LAZY=0 ... tests/engine` (lazy results off).
The strategy is `review/TESTS.md`.

| Directory | What | Style |
|---|---|---|
| `tests/atlas/` | every op x dtype x structure x view x storage against NumPy; `pytest tests/atlas` is the CI gate, `python -m tests.atlas.report` writes `review/atlas/ATLAS.md` | generated |
| `tests/engine/` | the engine suite, by subject: `test_conformance` (grid), `test_invariants`, `test_regressions` (one `test_F<id>` per fixed bug), linear algebra, matmul rules, views, scalar ops, bitwise, `test_interop` (NumPy in and out, indexing), `test_storage`, `test_persistence` (container robustness), `test_io_trace`, `test_streaming` (forced out-of-core), `test_warnings`, `test_api_surface`, `test_stress` (fuzzing, threads, leaks) | NumPy oracle |
| `tests/physics/` | causal sets, spacetimes, fields, plots (smoke tier for R1) | examples |
| `tests/blockmatrix/` | the BlockMatrix feature (construction, views, refinement, lazy thunks, persistence) | examples |
| `tests/*.cpp` | gtest: MemoryGovernor, IO accelerator, CoW, ParallelUtils, symmetric storage, CPU worker contract | unit |

Coverage-based retirement of the old `tests/python/` suite (TESTS.md §4) is complete; the
tools are `review/scripts/retire_analysis.py` and `cpp_retire_analysis.py`.

## Benchmarks

Location:
- `benchmarks/`

Benchmarks exist to compare:
- PyCauset vs NumPy for in-memory matrices,
- direct vs streaming paths,
- CPU vs GPU paths.

Key harnesses:
- `benchmarks/benchmark_numpy_parity.py`: **Critical.** Measures 1:1 throughput vs NumPy for import/export gates.
- `benchmarks/benchmark_io_smoke.py`: Measures basic save/load wall time and MB/s.

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

- `review/atlas/ATLAS.md` (operation x dtype x structure x view coverage; `tests/atlas/`)

When a checklist item changes status:
- ensure a corresponding test exists (or is added),
- ensure a benchmark script exists (or is updated).

### Stress and safety (`tests/engine/test_stress.py`)

Behaviour coverage cannot see: header fuzzing (random bytes never load), the payload on disk
after `save` (read back with plain file I/O), concurrent create / save / load / delete from
8 threads (verified on Linux; the old suite ran it on Windows only), and repeated 200 MB
allocations that must not accumulate RSS.
