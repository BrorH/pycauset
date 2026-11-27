# experimental/physics: causal sets, spacetimes, fields, plots (parked for R1)

**Status:** parked (maintainer decision, 2026-11-27). R1 is the linear-algebra engine, "NumPy
for causal sets"; the physics layer comes back after it. Nothing here is built, imported,
tested or documented with the package. This is the same arrangement as `experimental/cuda`.

## What is here

| Path | Was |
|---|---|
| `python/causet.py` | `pycauset.CausalSet`, `pycauset.causet(...)` |
| `python/spacetime.py` | `pycauset.spacetime` (Minkowski diamond / cylinder / box, de Sitter, FLRW, Schwarzschild, ...) |
| `python/field.py` | `pycauset.field` (`Field`, `CorrelatedField`, `ContinuumCorrelatedField`, `State`) |
| `python/synthetic.py` | `pycauset.synthetic` (chains, antichains, test orders) |
| `python/vis.py` | `pycauset.plot_embedding`, `plot_hasse`, `plot_causal_matrix`, `show` |
| `src/Sprinkler.cpp`, `include/Spacetime.hpp`, `include/Sprinkler.hpp` | native sprinkling and spacetimes (`src/causet/`, `include/pycauset/causet/`) |
| `src/bind_causet.cpp` | their bindings (`_native.MinkowskiDiamond`, `sprinkle`, `make_coordinates`, ...) |
| `tests/` | `tests/physics/` and the physics tests from `tests/engine` |
| `docs/` | the CausalSet, spacetime and field class pages, the function pages, and the guides (Causal Sets, Analyzing a Causal Set, Spacetime, Field Theory, Visualization) |

## What stayed in the engine

Everything that is linear algebra on causal matrices: `pycauset.causal_matrix` (a
`TriangularMatrix` of dtype `bool`), `TriangularMatrix.random`, the lazy `C ± C.T` closures,
`compute_k` / `compute_k_matrix`, and the link-matrix kernel (`_native._transitive_reduction`,
now `src/math/CausalMatrix.cpp`). A `.pycauset` file saved from a `CausalSet` still loads:
its payload is the causal matrix, which `load` returns (the spacetime metadata is not read).

## Open physics findings (review/FINDINGS.md)

- **F-105:** `MinkowskiDiamond(d > 2)` samples the cube `[0,1]^d` with a coordinate-wise
  order (a random d-dimensional order), not d-dimensional Minkowski space, without a warning.
  Decide before bringing it back: raise for d > 2, warn, or implement the real sampler.
- PLAN 4.7: one Minkowski implementation (Python and native duplicate each other;
  `tests/test_minkowski_paths_agree.py` pins that they agree).

## Bringing it back

1. Move the Python modules back to `python/pycauset/` and restore the imports and exports in
   `__init__.py` (`field`, `spacetime`, `synthetic`, `CausalSet`, `causet`, `plot_*`, `show`,
   and the field classes) and the `CausalSet` branches of `_internal/persistence.py`
   (`save` wrote `object_type`, `spacetime` and `seed` metadata; `load` rebuilt the object).
2. Move the C++ back (`src/causet/`, `include/pycauset/causet/`, `src/bindings/bind_causet.cpp`),
   re-add both to `CMakeLists.txt` and `bind_causet_classes(m)` to `src/bindings.cpp`.
3. Move the tests to `tests/physics/` and add it to `testpaths` in `pyproject.toml`; move the
   docs back and restore their mkdocs nav entries.
4. Resolve F-105 and PLAN 4.7 first.
