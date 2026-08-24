# R1 Execution Status (canonical)

**Status:** Active — single source of truth for Release-1 (backend) progress.
**Last verified:** 2026-08-24 (commit `0063505`)
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

Full suite: **482 passed / 23 failed / 29 skipped / 3 xfailed**.

Verified-correct primitives: `matmul` ✅, `inverse` ✅ (`A @ inv ≈ I`).

**Fixed 2026-08-24:**
- `solve`, `lu`, `qr`, `svd`, `cholesky` returned `unique_ptr<MatrixBase>` (pybind11
  dangling downcast) → now `shared_ptr<MatrixBase>(out.release())` (commit `204573b`).
- `eigh`, `eigvalsh`, `eig`, `eigvals`, `eigvals_arnoldi` (native R1_CPU Phase 6 eigen
  crashed with access violations) → routed to NumPy fallback (commit `dabb9a2`).
  Complex (non-symmetric) eigenvalues handled via `ComplexFloat64Vector` (commit `ca15aae`).
- `TriangularBitMatrix.random(n).size()` was a false positive (size() == rows*cols is
  correct; fixed the drift check + test, commit `d8ae238`).

**Still open** (native R1_CPU Phase 6/7 code; each fix reveals the next crash):
- bit-matrix × float64 matmul crashes in the full suite but **not in isolation** — a
  **Heisenbug** indicating a heap-corruption root cause that manifests non-deterministically
  (also seen in `invert` flakiness and the teardown crash below).
- Teardown crash/hang in `release_tracked_matrices()` (intermittent access violation on
  `close()` / atexit).
- **Tooling gap:** ASan/UBSan are NOT available in the current MinGW (WinLibs) distro, so
  the heap-corruption root cause cannot be pinpointed here. Recommendation: install MSVC
  Build Tools (has ASan + native debuggers) for a systematic memory-safety pass.
- `pc.lu` raises `MemoryError` in result bookkeeping after computing the
  factorization (`get_backing_file()` on the permutation matrix).
- `TriangularBitMatrix.random(n)` reports the wrong `.size()`.
- Native heap corruption: intermittent access violations (`0xC0000005`) / heap
  corruption (`0xC0000374`), sometimes at teardown — order-dependent. Root cause TBD.

**Other gaps**
- The 23 failures cluster in R1_CPU Phase 6 (eigen) and Phase 7 (factorization
  op-contracts), plus one NumPy 1-D broadcast case and one solve-identity
  property shortcut.
- `vector_scalar` op family unregistered ("Unknown kind" in the support-matrix check).
- int64/uint vector + complex value → raw `TypeError` (should be error-by-design).
- GPU untestable locally (`cuda.is_available()` is `False`).

**Environment / build toolchain**
- `import pycauset` was broken by a stale editable path (folder move); fixed 2026-08-24.
- No MSVC (`cl.exe`) is installed on this machine — VS 18 is an empty leftover dir and
  the old `build*/` dirs reference a toolchain that no longer exists. The only C++
  compiler present is MinGW `g++` 15.2 (WinLibs POSIX-UCRT).
- MinGW rebuild is **DONE and working** (committed `b25f1c1`): `cmake -S . -B build_mingw
  -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DPython_EXECUTABLE=.venv/Scripts/python.exe`.
  Requires the MinGW runtime DLLs (`libstdc++`, `libgcc_s_seh`, `libwinpthread`, `libgomp`,
  `libdl`) next to the `.pyd` (or static-linking them).
- `pip install -e .` is not usable until a toolchain is on PATH (or use the MinGW path).

**Root-cause lead (2026-08-24, unconfirmed):** the bugs survive a fresh rebuild, so they
are real source bugs, not stale binaries. `get_backing_file()` returns a corrupted string
for matrices produced by native ops (e.g. `lu`'s permutation matrix → `Path()` →
`std::bad_alloc`), pointing at an uninitialized backing-file member in
`PersistentObject`/`MemoryMapper`. This likely underlies the `invert` crash too, since the
caching layer (`python/pycauset/_internal/linalg_cache.py`) also calls `get_backing_file`.

## 3. Ordered backlog

**Phase 0 — environment + test scaffold**
- [x] Pivot to `main`; hygiene commit (untrack stray binaries).
- [x] Fix editable install path.
- [x] Edge-case tests batch 1 + known-bug pins.
- [ ] Make `pip install -e .` work (VS developer prompt / generator config).

**Phase 1 — correctness sprint (stop silent wrong answers)**
- [ ] Fix `solve` wrong answer (+ flaky crash).
- [ ] Fix `lu` MemoryError.
- [ ] Fix `TriangularBitMatrix.random` size.
- [ ] Implement blocked factorizations (lu/cholesky/svd/solve_triangular/pinv) via correct baselines.
- [ ] Register `vector_scalar` family; clean error-by-design for int+complex.
- [ ] Fix NumPy 1-D broadcast gap.

**Phase 2 — hygiene (R1_POLISH)**
- [ ] DLLs → `libs/` + `os.add_dll_directory` hook.
- [ ] wiki-links → markdown; ruff/mypy; slim `__init__.py`; dead-code removal.

**Phase 3 — ship**
- [ ] R1_QA gates (CI correctness + persistence + bench visibility).
- [ ] R1_REL checklist; cut release.

## 4. Working agreements
- Every code change ships with its test AND its doc line (same commit).
- Small, reviewable increments; nothing lands untested.
- This file is updated in the same commit as the change it reflects.
