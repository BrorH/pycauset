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

**Correctness — fixed and verified:**
- `solve`/`lu`/`qr`/`svd`/`cholesky` — was the flagship silent-wrong-answer; root cause
  `unique_ptr<MatrixBase>` → `shared_ptr` (commit `204573b`).
- `eigh`/`eigvalsh`/`eig`/`eigvals`/`eigvals_arnoldi` → NumPy fallback, incl. complex
  (non-symmetric) eigenvalues via `ComplexFloat64Vector` (`dabb9a2`, `ca15aae`).
- `TriangularBitMatrix.random(n).size()` — false positive; `size()==rows*cols` is correct
  (`d8ae238`).
- Dead overridden `lu`/`cholesky`/`svd` stubs removed (`17ab757`).

**Correctness — still open:**
- **Heap-corruption Heisenbug** (the main blocker): bit-matrix × float64 matmul crashes in
  the full suite but not in isolation; also seen as `invert` flakiness and a teardown crash
  in `release_tracked_matrices()`. Root cause is an out-of-bounds write — needs ASan.
- `vector_scalar` op family unregistered ("Unknown kind" in support-matrix check).
- int64/uint vector + complex value → raw `TypeError` (should be error-by-design).
- NumPy 1-D broadcast gap (one test).

**Verification ceiling:** the full suite still crashes mid-run at the Heisenbug, so there
is no clean full-suite count yet. Targeted suites (edge cases, factorizations, eigen) pass.

**Environment:** MinGW build works (committed `b25f1c1`) but is **CPU-only** (`ENABLE_CUDA=OFF`
— MinGW can't compile `.cu` kernels). The machine **does** have a GPU: GTX 1060 6GB, driver
582.53, CUDA toolkit 12.6. MSVC Build Tools now installed → rebuild with `-DENABLE_CUDA=ON`
to enable + verify the GPU path (R1_GPU / SRP-3).

## 3. Ordered backlog

**Phase 0 — environment + test scaffold**
- [x] Pivot to `main`; hygiene commit; editable-path fix; edge-case tests + bug pins.
- [ ] MSVC toolchain (in progress — user installing Build Tools).

**Phase 1 — correctness sprint**
- [x] `solve`/`lu`/`qr`/`svd`/`cholesky` (unique_ptr → shared_ptr).
- [x] eigen ops → NumPy fallback.
- [x] `TriangularBitMatrix.random` (false positive).
- [ ] Heap-corruption root cause (needs ASan) — bit-matmul / `invert` / teardown.
- [ ] `vector_scalar` registration; int+complex error-by-design; NumPy 1-D broadcast.

**Phase 2 — hygiene (R1_POLISH)**
- [ ] DLLs → `libs/` + `os.add_dll_directory` hook.
- [ ] wiki-links → markdown; ruff/mypy; slim `__init__.py`; remaining dead code.

**Phase 3 — ship**
- [ ] R1_QA gates (CI correctness + persistence + bench visibility).
- [ ] R1_REL checklist; cut release.

## 4. Working agreements
- Every code change ships with its test AND its doc line (same commit).
- Small, reviewable increments; nothing lands untested.
- This file is updated in the same commit as the change it reflects.
