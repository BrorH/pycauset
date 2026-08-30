# PyCauset TODO

The forward-looking work list. Release 1 (R1) shipped the correctness-focused linear
algebra foundation; this file tracks everything that remains after R1. The original
sequence-based R1 roadmap is retained at the bottom for reference, and the live R1
status lives in `R1_EXECUTION.md`.

## What's next (post-R1)

### Immediate: finish the R1 release
- [x] CI green on the 3-OS matrix (Ubuntu/macOS/Windows + Linux ASan, 2026-08-28)
- [ ] Linux build verified by maintainer (CI green now; manual spot-check optional)
- [ ] Tag `v0.6.1` and push (setuptools_scm picks up the tag as the version)
- [ ] Publish to PyPI and sanity-check `pip install pycauset` in a fresh venv
- [ ] Docs: convert wiki-links to markdown, finish API reference completeness
- [ ] API lock: mark `_internal` as private (`__all__` is already curated)

### Known issues (bugs, fix post-R1, do not forget)
- **Teardown hang in `release_tracked_matrices()`**, root-caused and fixed
  (R2_HARDEN, 2026-08): the `MemoryGovernor`/`ComputeContext`/`ThreadPool`/
  `OpRegistry` singletons were Meyers statics whose destruction order is undefined
  during interpreter finalization, so `PersistentObject` destructors and `close()`
  calling `instance()` could hang or crash. They are now heap singletons that are
  intentionally never destroyed (destructors are empty or the OS reclaims the
  resources at process exit), eliminating the whole class of teardown-ordering
  hazards. The `release_tracked_matrices()` finalization skip is retained as
  defense-in-depth (the OS reclaims mappings anyway).
- **Dead code / deprecated-feature sweep**, in progress (R2_HARDEN): removed the
  import-time-skipped `test_pauli_jordan_spectrum.py` (referenced the removed
  `.eigenvalues()` API) and the stale `*.dll.stale` build artifacts. `test_skew.py`
  and `test_skew_comprehensive.py` were re-enabled once the native `eigvals_skew`
  (R2_CATALOG skew eigensolver) landed in R2.2.

### Resolved (R1)
- **Constructor segfault / heap-corruption heisenbug (Linux SIGSEGV, macOS SIGBUS)**:
  root cause was `PersistentObject`'s destructor never calling
  `MemoryGovernor::unregister_object()`, so the governor LRU accumulated dangling
  `PersistentObject*` entries and `evict_until_fits()` dereferenced them via
  `spill_to_disk()` under memory pressure (ASan: heap-buffer-overflow in
  `PersistentObject::spill_to_disk()`). Fixed by unregistering in
  `~PersistentObject()`.

### Deferred optimization (continuous post-R1 program)
- Achieve >= 0.90x NumPy throughput for every op (the "never slower than NumPy" bar).
- GPU parity: CUDA 12.6 + VS 2022 for the GTX 1060 (Pascal); CUDA 13 dropped it.
- Streaming-everything: enable every out-of-core path.
- SRP-2 "Causal Math Optimization Catalog" (triangularity, Neumann series, property abuse).
- Eigen-cache persistence to the `.pycauset` container (avoid recompute on reload).
- macOS wheels: build OpenBLAS/libomp from source against a fixed older deployment
  target (currently pinned to macos-15 and bounded by Homebrew).

### Physics (Release 2)
- 100GB propagator matrix K (capstone large-scale experiment)
- Pauli-Jordan function i*Delta
- Curved spacetimes (Schwarzschild / de Sitter)
- User-defined spacetimes
- A user profiler (RAM/GPU/CPU) to drive automatic optimization

### Hygiene
- Remove dead code and legacy eager-evaluation paths
- Slim `__init__.py`
- ruff/mypy incremental cleanup (E/I/UP style rules), **in progress**: `ruff check --fix`
  applied import-sorting (I001) across `python/pycauset`. Remaining are ~295 mechanical
  findings: `E501` (docstring/line >100 chars, cosmetic) and `UP006`/`UP045`
  (`Tuple`→`tuple`, `Optional[X]`→`X | None`, `List`→`list`, need `--unsafe-fixes`;
  safe on the actual py3.10+ runtime but flagged against the stale `py38` target).

---

## R1 is complete

The R1 roadmap and node details now live in `internals/plans/R1_EXECUTION.md`, and the per-node planning docs are under `internals/plans/archive/`. They are history; this file only tracks forward-looking work.
