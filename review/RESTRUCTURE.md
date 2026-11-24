# Restructure & Cleanup Proposal (for discussion)

**Status:** DRAFT. Nothing here has been executed. Items marked **DECISION** need
the maintainer's call before anyone acts on them. Record each answer inline
(`**Decided (date):** ...`) so later agents don't re-litigate it.

Ground rules (from the maintainer, 2026-11-24):
- Pre-alpha: **no backwards compatibility**. When something goes, it goes
  completely: code, tests, docs, plan references.
- Git is the archive. We don't keep "archive/" folders or legacy shims.
- Moving files between directories is fine if the result is cleaner.
- Every moved or merged piece must be reflected in `review/ARCHITECTURE.md` and the
  user/dev docs in the same change. Undocumented code counts as lost code.
- GPU acceleration is **on hold**: don't invest in it, don't let it block anything.

---

## 1. Principles for the target layout

1. **One place per concept.** No component exists twice: two StreamingManagers,
   properties in C++ flags *and* a Python dict, Minkowski in Python *and* C++,
   operators in pybind overloads *and* Python wrappers.
2. **Names describe what, not when.** No `R1_`, `R2_`, `phase_b`, `phase3_4`,
   "Phase 3" in file names, test names or comments.
3. **Files small enough to review.** Rough cap of ~800 lines per source file.
   Today: `CpuSolver.cpp` 5,867, `bind_matrix.cpp` 4,719, `__init__.py` 2,416.
4. **The map is part of the code.** Each top-level source directory gets a short
   `README.md` saying what lives there and what doesn't. ARCHITECTURE.md links them.

## 2. Proposed moves, by area

### 2.1 Repository root
| now | proposal |
|---|---|
| `mockups/spacetime_studio_mockup.html` | delete (an idea, not code; note it on the roadmap if wanted) |
| `scripts/make_r1_gallery.py` | move to `tools/make_gallery.py` |
| `tools/` (mix of ps1 build scripts, audits, diagnostics) | keep `tools/`, split into `tools/dev/` (audits, checks) and `tools/release/`; delete one-off diagnostics (`macos_hang_diagnose.py`) once its bug is closed |
| `demos/` | keep (user-facing), and run it in CI as a smoke test |
| `benchmarks/` (11 scripts, overlapping) | consolidate into one harness plus scenarios; delete `benchmark_numpy_parity.py`, `benchmark_io_smoke.py`, `bench_coverage.py` if superseded (verify first) |
| `BENCHMARKS.md`, `CHANGELOG.md`, `THIRD_PARTY_NOTICES.md` | keep at root |
| `review/` | this workspace (see §5) |

### 2.2 C++ (`include/pycauset`, `src/`)
Keep the conventional `include/` + `src/` split. Changes inside it:

- **Split `CpuSolver.cpp`** by op family, behind the same class:
  `cpu/matmul.cpp`, `cpu/elementwise.cpp`, `cpu/vector_ops.cpp`,
  `cpu/reductions.cpp` (sum/trace/norm/det), `cpu/lapack.cpp`
  (inverse/solve/qr/lu/svd/cholesky), `cpu/eigen.cpp` (eigh/eig/arnoldi/skew),
  `cpu/simd/*.hpp` (SIMD kernels and cpuid dispatch). Pure moves first, then remove
  the duplicated dtype-dispatch ladders with one templated dispatcher.
- **Split `bind_matrix.cpp`**: `bind_matrix_classes.cpp` (class registration per
  dtype), `bind_matrix_numpy.cpp` (buffer protocol, from/to NumPy),
  `bind_matrix_operators.cpp` (one operator table, see DECISION 1).
- **Delete**: `WorkloadPartitioner.{hpp,cpp}`, `src/main.cpp`, the
  `#pragma message`, the commented-out CMake targets.
- **`PersistentObject`** moves into `namespace pycauset`.
- **`MatrixType` enum**: separate structure from dtype (DECISION 4).
- **CUDA** (`src/accelerators/cuda/`, `bind_cuda.cpp`, `pycauset/cuda.py`): DECISION 3.

### 2.3 Python (`python/pycauset/`)
- **`__init__.py` becomes a pure facade** (target < 300 lines): imports and
  `__all__` only. Factories move to `_internal/factories.py`, linalg front-ends to
  `ops`, I/O to `persistence`, debug mode to `_internal/debug.py`.
- **Group `_internal/` (30 flat modules) into subpackages**:
  ```
  _internal/
    native/       native.py, native_setup.py, native_threshold.py, gpu_hint.py
    core/         dtypes.py, coercion.py, formatting.py, warnings.py, runtime.py
    ops/          ops.py split into linalg.py, eig.py, decomp.py, elementwise.py
    properties/   properties.py (after DECISION 2 shrinks it)
    storage/      persistence.py, big_blob_cache.py, linalg_cache.py,
                  export_guard.py, io_observability.py, streaming_manager.py
    blocks/       blockmatrix.py, submatrix_view.py, dense_python.py, thunks.py
  ```
- **Delete**: `lazy_ufunc.py`, `interop.patch_interop`, `_streaming_eigh`,
  `_storage.StorageRegistry`/`tracked_temp_files`, unreachable code in
  `ops.eigvals`. Decide whether `storage_debug.py` and `support_matrix.py` are dev
  tools (move to `tools/`) or library features.
- **Remove the monkeypatch wrapper stack** (`patching.py` plus the operator wrappers
  in `properties.py`). See DECISION 1.
- **Silent `except: pass`** (126 sites): each one either becomes a specific
  exception type or gets a comment explaining why swallowing is correct.
- **`.pyi` stub**: generated, or deleted (F-007).

### 2.4 Tests
```
tests/
  conftest.py           one place for fixtures (tmp backing dir, memory threshold,
                        NumPy-parity helpers). NO sys.path hacks: tests run against
                        the installed / editable package.
  engine/               dtypes/, construction/, elementwise/, matmul/, linalg/,
                        storage/, persistence/, properties/, blocks/, lazy/
  physics/              causet/, spacetime/, field/, synthetic/, vis/
  parity/               generated NumPy-parity grids (from review/probes)
  docs/                 runs every user-facing doc snippet
  cpp/                  gtest sources (all wired into CMake, all run in CI)
  slow/ or a marker     large-N, out-of-core, stress
```
- Rename away milestone names (`test_blockmatrix_phase_c_views.py` →
  `engine/blocks/test_views.py`, `test_field_r2.py` → `physics/field/test_field.py`).
- Merge near-duplicates (e.g. `test_inverse_{correctness,cpu,cpu_block,extensive}.py`,
  `test_numpy_{integration,interop,interop_ergonomics,conversion_policy}.py`,
  `test_skew{,_comprehensive}.py`).
- Delete `tests/manual/`, `tests/stability/` (or turn them into `slow` tests),
  `tests/python/debug_load.py`, `phase3_4_verification.py`.
- GPU-only tests: park them with the CUDA code (DECISION 3).
- `tests/BUG_LOG.md` moves into the docs (or merges with `review/FINDINGS.md`).

### 2.5 Documentation (`documentation/`)
Today: `guides/` (user), `docs/` (API reference, 96 function pages + class pages),
`internals/` (design), `dev/` (developer notes), `project/` (philosophy, protocols,
plans), `internals/plans/` and `internals/plans/archive/` (18 historical plans).

Proposal:
- **Delete** `internals/plans/archive/` (git has it) and the finished
  R1/R2 plans. Keep one `ROADMAP.md`.
- **Merge** `internals/` and `dev/` into one `internals/` (developer docs).
  `review/ARCHITECTURE.md` becomes its front page once the review is complete.
- **API reference**: DECISION 5.
- Every python code block in user docs must run; enforced in CI by the doc-snippet
  test. Signature listings get a non-python fence.

## 3. DECISIONS needed from the maintainer

1. **Where do operator semantics live?** Today they're split between ~8 pybind
   overloads per operator and two layers of Python monkeypatch wrappers.
   *Recommendation:* C++ is the single source of truth for arithmetic and result
   types (fast, one place). Python adds nothing per-op, apart from a single
   registration hook for temp-file tracking implemented in C++ or via pybind's
   return-value policy. Properties propagation moves into C++ alongside
   `PropertyFlags` (see 2).
2. **Properties system:** C++ flags and the Python dict both exist. *Recommendation:*
   C++ owns the flags and the propagation rules; Python exposes a read-only view
   plus explicit setters.
3. **GPU/CUDA code while on hold:** (a) leave in place, unbuilt and untested; (b) move
   to `experimental/cuda/` with its tests; (c) delete now and recover from git
   later. *Recommendation:* (b). It keeps the tree honest about what's maintained
   without throwing away work.
4. **Class naming and type enum:** rename `IntegerMatrix` → `Int32Matrix`,
   `FloatMatrix` → `Float64Matrix`, `IntegerVector`/`FloatVector` likewise, and split
   `MatrixType` into Structure × DType. Breaking, but we're pre-alpha.
   *Recommendation:* yes, in one sweep.
5. **API reference docs:** hand-maintained pages (96+, drifting) vs generated from
   docstrings (mkdocstrings), with hand-written guides kept. *Recommendation:*
   generate, and make the docstrings the source of truth.
6. **Python floor:** `requires-python >= 3.8` but CI tests only 3.12 and the code
   uses 3.10+ idioms. *Recommendation:* `>=3.10`, CI on 3.10 and 3.13.
7. **bool semantics:** NumPy (bool + bool = OR) or counting semantics
   (bool + bool = int, which suits causal matrices)? Either way it must be one
   documented rule (F-013).
8. **Export guard and spill dir UX** (F-016, F-017): guard by size only (not by
   "is on disk")? Default spill dir to the system temp dir instead of `./.pycauset`?

## 4. Sequencing
Don't restructure before the safety net exists (PLAN phase 1). Moves that change no
behaviour (file splits, renames) go in their own commits, separate from logic
changes, so `git log --follow` and reviews stay readable.

## 5. The `review/` workspace itself
It stays for the duration of the overhaul. Afterwards, ARCHITECTURE.md moves into
`documentation/internals/`, FINDINGS.md becomes the bug log, and `review/` is deleted.
