# Roadmap

What PyCauset is for, release by release. The working plan with every task and its status is
the review workspace (`review/PLAN.md`; findings in `review/FINDINGS.md`); this page is the
summary.

## Release 1: the engine (in progress)

"NumPy for causal sets": a NumPy-like matrix engine whose structure classes (bit-packed
causal matrices, triangular, symmetric, Hermitian, diagonal, identity) exist for every dtype,
with NumPy as the reference for every value and dtype.

In scope:

- Every operation for every dtype, structure and view, checked against NumPy (the operation
  atlas: about 96,000 combinations, 0 wrong).
- Linear algebra through LAPACK with NumPy's conventions.
- Storage in RAM or memory-mapped files (decided when an object is made, by size and free
  RAM), and `.pycauset` snapshots.
- Causal sets: `pc.causet`, sprinkling into the d-dimensional Minkowski diamond, the order
  methods (links, chains, layers, intervals, past / future), Myrheim-Meyer, synthetic orders,
  save / load.
- Lightweight plots (`pc.vis`).

Still to do before the release (review PLAN):

- Cleanup: splitting the largest files, the remaining review findings (`review/FINDINGS.md`).
- Speed: at least 0.9x NumPy in memory across the atlas (`review/atlas/TIMING.md`). Today
  most dense operations are there; small-call overhead and some structured and complex32
  cases are not.
- Features: the derived-value cache (scalar and vector results, saved with the file, done;
  matrix-valued results next, `review/features/DERIVED_CACHE.md`),
  lazy expressions (`LAZY.md`), value bounds for exact integer result dtypes.
- Release mechanics: tag a version, publish wheels for Linux, macOS and Windows, check
  `pip install pycauset` in a fresh environment.

## Release 2: physics on causal sets (parked)

Fields on causal sets (retarded and advanced propagators, the Pauli-Jordan function,
Sorkin-Johnston vacuum), a spacetime library beyond flat Minkowski (curved and user-defined
spacetimes) and large propagator experiments. The code and its design documents are in
`experimental/physics/` (plans in `experimental/physics/docs/plans/`); they come back into the
package after Release 1.

## Later

- A release about visualization.
- GPU (CUDA) acceleration: the device code is parked in `experimental/cuda/`; before it
  returns, GPU failures must raise instead of printing and falling back (review F-116).
- Out-of-core benchmarks on a real disk at sizes far beyond RAM.
