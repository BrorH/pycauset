# Testing & Benchmarks

This page documents how to validate correctness and performance.

## Test layers

Run everything with `PYTHONPATH=python python -m pytest -q -p no:cacheprovider` (the
`testpaths` in `pyproject.toml`), plus `PYCAUSET_LAZY=0 ... tests/engine` (lazy results off).
The strategy is `review/TESTS.md`.

| Directory | What | Style |
|---|---|---|
| `tests/atlas/` | every op x dtype x structure x view x storage against NumPy; `pytest tests/atlas` is the CI gate, `python -m tests.atlas.report` writes `review/atlas/ATLAS.md` | generated |
| `tests/engine/` | the engine suite, by subject: `test_conformance` (grid), `test_invariants`, `test_regressions` (one `test_F<id>` per fixed bug), linear algebra, matmul rules, views, scalar ops, bitwise, `test_interop` (NumPy in and out, indexing), `test_storage`, `test_persistence` (container robustness), `test_io_trace`, `test_streaming` (forced out-of-core), `test_warnings`, `test_api_surface`, `test_stress` (fuzzing, threads, leaks), `test_causet` and `test_plots*` (causal sets, sprinkling, plots), `test_doc_snippets` (the code in the guides runs) | NumPy oracle |
| `tests/blockmatrix/` | the BlockMatrix feature (construction, views, refinement, lazy thunks, persistence) | examples |
| `tests/*.cpp` | gtest: MemoryGovernor, IO accelerator, CoW, ParallelUtils, CPU worker contract | unit |

Coverage-based retirement of the old `tests/python/` suite (TESTS.md §4) is complete; the
tools are `review/scripts/retire_analysis.py` and `cpp_retire_analysis.py`.

## Benchmarks

`BENCHMARKS.md` (repository root) lists the scripts and the current numbers. The two that
matter most: `python benchmarks/parity.py` (headline ops; CI runs it, report-only) and
`python -m tests.atlas.timing [op ...]` (every atlas cell against NumPy, written to
`review/atlas/TIMING.md`).

## Protocol

- Correctness first: tests must pass before trusting benchmarks.
- Benchmarks should be run with stable conditions:
  - consistent seeds,
  - clear dtype,
  - documented hardware.

## Link to optimization tracking

The authoritative checklist for dtype/op coverage and readiness gates is:

- `review/atlas/ATLAS.md` (operation x dtype x structure x view coverage; `tests/atlas/`)

A new operation enters the atlas (correctness) and its timing (speed) together
([[project/protocols/Adding Operations|Adding Operations]]).

### Stress and safety (`tests/engine/test_stress.py`)

Behaviour coverage cannot see: header fuzzing (random bytes never load), the payload on disk
after `save` (read back with plain file I/O), concurrent create / save / load / delete from
8 threads (verified on Linux; the old suite ran it on Windows only), and repeated 200 MB
allocations that must not accumulate RSS.
