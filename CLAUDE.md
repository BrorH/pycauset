# PyCauset: notes for agents and contributors

PyCauset is "NumPy for causal sets": a C++20 engine (pybind11, OpenBLAS/LAPACK, OpenMP) with
specialised matrix types (bit-packed causal matrices, triangular, symmetric, diagonal, …) for
every dtype, behind a NumPy-like Python API. NumPy is the oracle for values and dtypes, never a
backend. Pre-alpha: no backwards compatibility; removed things are removed with their docs.

## Read before changing code
- `documentation/internals/Architecture.md`: the architecture map (keep it true).
- `review/README.md`: the ongoing review (plan, findings, log).
- `documentation/project/protocols/Adding Operations.md`: **the procedure for adding or
  changing an operation** (entry function + structure rules + normaliser + kernels with
  contracts; atlas and timing gates). Follow it.
- `documentation/project/protocols/Adding Types.md`: the procedure for adding a structure or
  dtype.
- `documentation/project/protocols/Pre-alpha Policy.md`.

## Rules that are easy to break
- One definition of view semantics: `include/pycauset/math/ViewValue.hpp`. Never re-derive it.
- No silent fallbacks, no NumPy behind a PyCauset call, no result caching outside the
  derived-value cache (`include/pycauset/math/DerivedCache.hpp`; spec `review/features/DERIVED_CACHE.md`). Structure is the
  class (there is no assertion/flag store). Guesses (sampling, heuristics) may warn, never decide.
- One code path per operation; one templated kernel per family, not per dtype.

## Build and test
```
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release   # once; wheels are Release (-O3)
cmake --build build/cpp --target _pycauset -j1        # -j2 runs out of memory
# Do not edit sources while a build runs: a compile that started before the edit finishes
# after it, its object is newer than the source, and make keeps the stale object.
cp build/cpp/_pycauset*.so build/cpp/libpycauset_core.so python/pycauset/
PYTHONPATH=python python -m pytest tests/engine tests/blockmatrix -q -p no:cacheprovider
PYCAUSET_LAZY=0 PYTHONPATH=python python -m pytest tests/engine -q -p no:cacheprovider
PYTHONPATH=python python -m pytest tests/atlas -q        # every op x dtype x structure x view
PYTHONPATH=python python -m tests.atlas.report [op ...]  # regenerate review/atlas/ATLAS.md
PYTHONPATH=python python -m tests.atlas.timing [op ...]  # speed vs NumPy, review/atlas/TIMING.md
```
