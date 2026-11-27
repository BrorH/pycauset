# experimental/physics: fields and the spacetime library (parked for R1)

**Status:** parked (maintainer decision, 2026-11-27). Nothing here is built, imported, tested
or documented with the package; the same arrangement as `experimental/cuda`.

**What R1 keeps in the package** (the tools to create causal sets and test the engine with):
`CausalSet` / `pc.causet`, `pc.spacetime.MinkowskiDiamond(d)` (the true diamond in any
dimension, one native implementation), the order methods (links, chains, layers, intervals,
past / future), the Myrheim-Meyer estimate, `pc.synthetic`, save / load, and lightweight plots (`pc.vis`: embedding, Hasse diagram,
causal matrix), and the engine's causal-matrix operations (`causal_matrix`, lazy `C ± C.T`, `compute_k`, the link-matrix
kernel).

## What is here

| Path | Contents |
|---|---|
| `python/field.py` | `pycauset.field` (`Field`, `CorrelatedField`, `ContinuumCorrelatedField`, `State`) |
| `python/spacetime.py` | the pre-R1 spacetime library: Python `Spacetime` base and registry, Minkowski cylinder / box, de Sitter, anti-de Sitter, FLRW, Schwarzschild, transformed / restricted / conformal / periodic spacetimes |
| `python/causet.py`, `python/synthetic.py` | the pre-R1 versions (Python-defined spacetimes); the package has the R1 versions |
| `include/Spacetime.hpp`, `src/bind_causet.cpp` | the native cylinder and box and their bindings |
| `tests/`, `docs/`, `scripts/` | their tests, doc pages and guides (Field Theory, Spacetime, Tutorials), the gallery script |

## Before bringing something back

- The native cylinder and box use the coordinate-wise placeholder for d > 2 that F-105 fixed
  for the diamond; give them the same treatment (sample the region uniformly, Minkowski
  causality, exact volume).
- One implementation per spacetime (PLAN 4.7): the Python class describes it; sampling and
  causality are native.
- Restore the exports in `__init__.py`, the CMake sources, the tests (`testpaths` in
  `pyproject.toml`), the doc pages and their mkdocs nav entries.
