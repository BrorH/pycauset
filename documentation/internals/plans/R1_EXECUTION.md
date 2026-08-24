# R1 Execution Status (canonical)

**Status:** Active — single source of truth for Release-1 (backend) progress.
**Last verified:** 2026-08-24
**Roadmap graph:** `TODO.md` · **SRP gates:** `SUPPORT_READINESS_FRAMEWORK.md`

> Updated ONLY when something is verified by running code. Never claims "done"
> unless tests pass. Where this file conflicts with a per-node status marker in
> an individual plan doc, this file wins.

## 1. Re-scoped Release-1 gate

R1 ships when the backend is **correct and trustworthy**, not maximally fast.

**Ship gate**
- Correct across the public surface — no silent wrong answers.
- Every op has an explicit support status (CPU/GPU/out-of-core), even if that
  status is "CPU-only, naive out-of-core".
- Known-broken things are documented as regression tests, not hidden.

**Deferred to a continuous post-R1 program**
- ≥0.90× NumPy throughput for every op.
- GPU parity for every op.
- Streaming-enable every out-of-core path.
- SRP-2 "Causal Math Optimization Catalog".

## 2. Measured state (2026-08-24)

**Full suite (MSVC build, no crash): 510 passed / 1 failed / 29 skipped.**
(Remaining failure: `test_eigen_caching.py::test_cache_persistence_across_load` — eigen cache not persisted to `.pycauset`; known, tracked in TODO.)

**Correctness — fixed and verified:**
- `solve`/`lu`/`qr`/`svd`/`cholesky` — was the flagship silent-wrong-answer; root cause
  `unique_ptr<MatrixBase>` → `shared_ptr` (commit `204573b`).
- `eigh`/`eigvalsh`/`eig`/`eigvals`/`eigvals_arnoldi` → NumPy fallback, incl. complex
  eigenvalues (`dabb9a2`, `ca15aae`); square-shape rejection (`90cee4c`).
- `TriangularBitMatrix.random(n).size()` — false positive (`d8ae238`).
- OpRegistry shared across DLLs (out-of-line `instance()`, `1841382`).
- `solve` property-as-gospel shortcuts restored (identity/zero/triangular, `09561e4`).
- NumPy 1-D broadcast (SIMD fast-path ignored shape → guarded, `90cee4c`).
- `load_matrix` alias; `eigh`/`eig` marked non-streaming routing (`cdb14eb`).
- Dead overridden stubs removed (`17ab757`).
- **"Heap-corruption Heisenbug" was MinGW-specific** — MSVC (with/without ASan) runs the
  full suite cleanly with zero ASan errors.

**Remaining (1 failure — Phase 6 *feature completeness*, not correctness):**
- `test_eigen_caching::test_cache_persistence_across_load`: eigen cache is not persisted
  to the `.pycauset` backing file (`matrix(backing_file=...)` + `sync()` doesn't write
  the file). Eigen-cache persistence is a Phase 6 caching feature — deferred.
- Intermittent **teardown hang** in `release_tracked_matrices()` (at exit; doesn't affect
  results).

**Environment:** MSVC Build Tools 2026 → canonical CPU build works (`build_msvc`).
**CUDA blocked by toolchain:** CUDA 13.0 (installed) has *dropped* Pascal (GTX 1060 = CC 6.1,
min is now CC 7.5). CUDA 12.6 still supports Pascal but does **not** support MSVC 2026
(nvcc's `cudafe++` crashes with VS-2026 headers; `-allow-unsupported-compiler` doesn't
help). → **CUDA build requires VS 2022 (MSVC 14.4x)** alongside the existing VS 2026.

## 3. Ordered backlog

**Phase 0 — environment + test scaffold**
- [x] Pivot to `main`; hygiene; editable fix; edge-case tests + bug pins.
- [x] MSVC toolchain installed; build working.

**Phase 1 — correctness sprint**
- [x] `solve`/`lu`/`qr`/`svd`/`cholesky`; eigen ops; `random` false-positive; OpRegistry; broadcast; property shortcuts.
- [x] Heap-corruption Heisenbug — resolved (was MinGW-specific; MSVC build is clean).
- [ ] Eigen-cache persistence to `.pycauset` (Phase 6, deferred).
- [ ] `vector_scalar` registration; int+complex error-by-design (lower priority).
- [ ] Teardown hang in `release_tracked_matrices()`.

**Phase 2 — hygiene (R1_POLISH)**
- [ ] DLLs → `libs/` + `os.add_dll_directory` hook.
- [ ] wiki-links → markdown; ruff/mypy; slim `__init__.py`; remaining dead code.

**Phase 3 — GPU + ship**
- [ ] Enable `ENABLE_CUDA=ON`; verify GPU routing/parity (R1_GPU / SRP-3).
- [ ] R1_QA gates (CI correctness + persistence + bench visibility).

**Phase 4 — Shipping readiness (R1_SHIP, production-release checklist)**
- [ ] **CPU baseline / AVX-512**: kernels use `_mm512_*` unconditionally → would crash
  (illegal instruction) on CPUs without AVX-512. Add runtime dispatch or build a baseline
  target + gated fast paths. (This is the same "silent crash" class we've been fixing.)
- [ ] **CUDA**: compile for target compute capabilities (GTX 1060 = CC 6.1); ensure clean
  CPU fallback when CUDA is absent; decide whether to bundle NVIDIA DLLs (EULA limits
  redistribution — likely require user-installed CUDA + dynamic load instead of shipping ~1.75GB).
- [ ] **Wheel portability**: verify Linux/macOS builds actually compile (GCC/Clang are
  stricter than MSVC — same `template`/`<cstring>`/SIMD-flag issues we fixed for MinGW).
- [ ] **API lock**: freeze the public `pycauset.*` surface; mark `_internal` private;
  version (setuptools_scm) + changelog + `.pycauset` format migration path.
- [ ] **CI test matrix** (3 OSes) + benchmark visibility; fix teardown hang.
- [ ] **Docs**: wiki-links → markdown, API reference completeness, install/GPU requirements.
- [ ] **License/attribution**: Eigen (MPL2), OpenBLAS (BSD), pybind11 (BSD), CUDA (proprietary).

**Phase 5 — release mechanics (R1_REL)**
- [ ] Release checklist; cut release.

## 4. Working agreements
- Every code change ships with its test AND its doc line (same commit).
- Small, reviewable increments; nothing lands untested.
- This file is updated in the same commit as the change it reflects.
