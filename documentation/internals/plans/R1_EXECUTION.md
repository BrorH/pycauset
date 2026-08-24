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

**Full suite (MSVC build, no crash): 487 passed / 21 failed / 29 skipped.**

**Correctness — fixed and verified:**
- `solve`/`lu`/`qr`/`svd`/`cholesky` — was the flagship silent-wrong-answer; root cause
  `unique_ptr<MatrixBase>` → `shared_ptr` (commit `204573b`).
- `eigh`/`eigvalsh`/`eig`/`eigvals`/`eigvals_arnoldi` → NumPy fallback, incl. complex
  eigenvalues (`dabb9a2`, `ca15aae`).
- `TriangularBitMatrix.random(n).size()` — false positive (`d8ae238`).
- Dead overridden stubs removed (`17ab757`).
- **"Heap-corruption Heisenbug" was MinGW-specific** — the MinGW build crashed on the
  bit-matrix matmul / `invert` / teardown, but the MSVC build (with and without ASan)
  runs the full suite cleanly with zero ASan errors. Root cause of the MinGW crashes is
  likely the `-march=native` SIMD codegen; the MSVC toolchain is the fix.

**Remaining (21 failures — R1_CPU Phase 6/7 *feature completeness*, not crashes):**
- ~13 "op not registered in OpRegistry" (trace/determinant/frobenius_norm/lu/qr/solve/svd
  op-contracts, eigen op-contracts).
- `load_matrix` missing (eigen caching test).
- `vector_scalar` op family unregistered; int64/uint+complex raw `TypeError`.
- NumPy 1-D broadcast gap; solve-identity property shortcut.
- Intermittent **teardown hang** in `release_tracked_matrices()` (at exit; doesn't affect
  test results).

**Environment:** MSVC Build Tools 2026 installed → **canonical build works**
(`build_msvc`). GPU present: GTX 1060 6GB + CUDA 12.6 toolkit; CUDA not yet compiled in
(`ENABLE_CUDA=OFF`) — to enable + verify R1_GPU.

## 3. Ordered backlog

**Phase 0 — environment + test scaffold**
- [x] Pivot to `main`; hygiene; editable fix; edge-case tests + bug pins.
- [x] MSVC toolchain installed; build working.

**Phase 1 — correctness sprint**
- [x] `solve`/`lu`/`qr`/`svd`/`cholesky`; eigen ops; `random` false-positive.
- [x] Heap-corruption Heisenbug — resolved (was MinGW-specific; MSVC build is clean).
- [ ] Register Phase 7 ops in OpRegistry (trace/determinant/frobenius_norm/lu/qr/solve/svd).
- [ ] `load_matrix`; `vector_scalar` registration; int+complex error-by-design; NumPy 1-D broadcast.
- [ ] Teardown hang in `release_tracked_matrices()`.

**Phase 2 — hygiene (R1_POLISH)**
- [ ] DLLs → `libs/` + `os.add_dll_directory` hook.
- [ ] wiki-links → markdown; ruff/mypy; slim `__init__.py`; remaining dead code.

**Phase 3 — GPU + ship**
- [ ] Enable `ENABLE_CUDA=ON`; verify GPU routing/parity (R1_GPU / SRP-3).
- [ ] R1_QA gates (CI correctness + persistence + bench visibility).
- [ ] R1_REL checklist; cut release.

## 4. Working agreements
- Every code change ships with its test AND its doc line (same commit).
- Small, reviewable increments; nothing lands untested.
- This file is updated in the same commit as the change it reflects.
