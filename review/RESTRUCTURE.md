# Restructure & Cleanup Proposal (for discussion)

**Status:** DRAFT proposal; technical decisions recorded in §3.1, product questions
open in §3.2. Nothing has been executed yet.

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
| `scripts/make_r1_gallery.py` | moved: `tools/make_gallery.py` (review 3.12 C-5) |
| `tools/` (mix of ps1 build scripts, audits, diagnostics) | keep `tools/`, split into `tools/dev/` (audits, checks) and `tools/release/`; delete one-off diagnostics (`macos_hang_diagnose.py`) once its bug is closed |
| `demos/` | keep (user-facing), and run it in CI as a smoke test |
| `benchmarks/` (11 scripts, overlapping; 3 kept after review 3.15) | consolidate into one harness plus scenarios; delete `benchmark_numpy_parity.py`, `benchmark_io_smoke.py`, `bench_coverage.py` if superseded (verify first) |
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
  `bind_matrix_operators.cpp` (one operator table, see §3.1 item 1).
- **Delete**: `WorkloadPartitioner.{hpp,cpp}`, `src/main.cpp`, the
  `#pragma message`, the commented-out CMake targets.
- **`PersistentObject`** moves into `namespace pycauset`.
- **`MatrixType` enum**: separate structure from dtype (§3.1 item 4).
- **CUDA** (`src/accelerators/cuda/`, `bind_cuda.cpp`, `pycauset/cuda.py`): moves to `experimental/cuda/` (§3.1 item 3).

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
    properties/   properties.py (after §3.1 item 2 shrinks it)
    storage/      persistence.py, big_blob_cache.py, linalg_cache.py,
                  export_guard.py, io_observability.py, streaming_manager.py
    blocks/       blockmatrix.py, submatrix_view.py, dense_python.py, thunks.py
  ```
- **Delete**: `lazy_ufunc.py`, `interop.patch_interop`, `_streaming_eigh`,
  `_storage.StorageRegistry`/`tracked_temp_files`, unreachable code in
  `ops.eigvals`. Decide whether `storage_debug.py` and `support_matrix.py` are dev
  tools (move to `tools/`) or library features.
- **Remove the monkeypatch wrapper stack** (`patching.py` plus the operator wrappers
  in `properties.py`). See §3.1 item 1.
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
- GPU-only tests: park them with the CUDA code (§3.1 item 3).
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
- **API reference**: generated from docstrings (§3.1 item 5).
- Every python code block in user docs must run; enforced in CI by the doc-snippet
  test. Signature listings get a non-python fence.

## 3. Decisions

**Roles (maintainer, 2026-11-24):** the maintainer is creative director; the
technical lead (the agent running the overhaul) makes technical calls and records
them here with a reason. The maintainer can veto any of them. Product questions
(what PyCauset *is* and *offers*) stay with the maintainer and are asked in plain
language.

### 3.1 Technical decisions (2026-11-24; items 1, 3, 4, 5, 7 confirmed by the maintainer)

Guiding principle (maintainer): *the whole module is used from Python, but the
engine and everything behind it is C++.* Python is a thin packaging layer.
1. **[Confirmed]** **Operator semantics live in C++ only.** One operator table in the bindings,
   no Python per-op wrappers. The monkeypatch stack (`patching.py` plus the
   operator wrappers in `properties.py`) is deleted. Temp-file registration
   happens in C++ at allocation. *Why:* one source of truth; removes the
   inconsistency behind F-001/F-002 and the overhead behind F-012.
2. **[Delegated to tech lead]** **Properties: C++ owns the flags and propagation rules**; Python exposes a view
   plus explicit setters. *Why:* two sources of truth today.
3. **[Confirmed]** **CUDA moves to `experimental/cuda/`** (code, bindings, Python facade, GPU
   tests), excluded from the default build, CI and docs nav. *Why:* keeps the
   maintained tree honest and out of agents' context, without losing the work.
4. **[Revised by tech lead, session 4; maintainer may veto]** **One Python class per
   structure; dtype is an attribute (NumPy model).** Classes: `DenseMatrix`,
   `TriangularMatrix`, `DiagonalMatrix`, `SymmetricMatrix`, `AntiSymmetricMatrix`,
   `IdentityMatrix`, `DenseVector`, `UnitVector`, each covering all 16 dtypes, with
   `A.dtype` (NumPy dtype name). C++: `MatrixType` became `Structure` (DENSE, TRIANGULAR,
   …); a causal matrix is TRIANGULAR with dtype bit.
   *Supersedes* the earlier confirmed plan of per-dtype class names (`Int32Matrix`,
   `TriangularInt16Matrix`, ~110 classes). *Why:*
   (a) lazy cast views (features/LAZY.md L1): `2 * C` is an int32 matrix backed by bit
   storage, which a dtype-in-the-class-name model can't express without copying;
   (b) value bounds (F1) pick small dtypes at run time; with per-dtype classes every
   combination needs a registered class, and a missing one surfaces as a bare
   `MatrixBase` (F-032). Structure classes make F-032 impossible by construction;
   (c) bind_matrix.cpp/bind_vector.cpp have 40 near-identical per-dtype class blocks
   (~3,000 lines) that collapse into one generic implementation per structure (4.3);
   (d) one API page per class instead of ~110.
   *User-visible change:* `isinstance(C, pc.TriangularBitMatrix)` →
   `isinstance(C, pc.TriangularMatrix) and C.dtype == "bool"`.
   *Mechanism:* non-template structure bases in C++ plus a pybind11
   `polymorphic_type_hook<MatrixBase>` that maps every object to its structure class
   from its `Structure` tag. Every packed (anti)symmetric object derives both markers, so
   the hook's pointer cast is always valid.
5. **[Confirmed]** **API reference generated from docstrings** (mkdocstrings), one
   page per public function and class (like NumPy's reference); guides stay
   hand-written. Maintainer's bar: short, direct and comprehensive. Page count isn't
   a metric; content quality is. **No filler or AI slop** (see PLAN Phase 5 style
   rules). *Why:* 96 hand pages already drift (F-008).
6. **Python ≥ 3.10**; CI on 3.10 and 3.13. *Why:* the code already needs it.
7. **[Confirmed]** **Export guard by size only.** Small disk-backed results convert to NumPy
   freely; the guard trips only above the size ceiling. *Why:* spilling is
   automatic, so the user can't predict it (F-016).
8. **[Maintainer decision, reverses the tech lead's proposal]** **Spill directory stays
   `./.pycauset`** (overridable with `set_backing_dir`). *Why:* PyCauset keeps full
   control over the lifetime of its files; the OS temp dir would cede that. Action:
   document it prominently; keep F-024 (stray `_vals` files) as a bug regardless.
9. **The old test suite is retired only after the new one proves equal or better
   coverage** (TESTS.md §4). *Why:* the old tests are the only written record of
   intended behaviour in some corners.
10. **[Maintainer decision 2026-11-24, reverses the tech lead's proposal to drop
   complex_float16]** **All 16 dtypes stay**: bit/bool, int8–64, uint8–64,
   float16/32/64, complex16/32/64. *Why (maintainer):* small dtypes are the
   under-promotion mantra ("massive matrices that just work"), so every dtype must
   work with every other. int64 is the ceiling (no int128). **Add none** (bfloat16,
   float128 rejected as creep). Consequence: complex16 exports to NumPy as complex64
   (NumPy has no complex32); document it. The hand-written `ComplexFloat16Matrix/Vector`
   classes should be folded into the generic templates during Phase 4 if feasible.
11. **Structures for R1:** keep all current structures for now; each is judged
   individually in Phase 3 by coverage and usefulness (DEADCODE Tier B), not cut
   wholesale.

### 3.2 Product questions for the creative director (open)
A. **[Decided by maintainer 2026-11-24] Scope for R1:** all 16 dtypes (§3.1 item 10).
   **Keep and test:** skew-symmetric eigensolvers (`eigvals_skew`, `eig_skew`) and
   `eigvals_arnoldi` ("definitely needed"). **Defer out of R1 and remove:**
   `precision_mode` / `set_precision_mode` / `get_precision_mode` and the C++
   thread-local precision mode (the default float under-promotion stays; the
   under-promotion warning should advise casting inputs to float64). Other rarely used
   features: tech lead judges per feature in Phase 3.
B. **[Decided by maintainer 2026-11-24]** bit/bool `A + B` **adds into integers**
   (counting), *not* logical OR. Multiplication keeps counting paths. Documented as a
   deliberate NumPy divergence. (F-013)
C. **[New feature requested by maintainer 2026-11-24]** value bounds and
   bounds-aware dtype selection → `review/features/VALUE_BOUNDS.md`.

## 4. Sequencing
Don't restructure before the safety net exists (PLAN phase 1). Moves that change no
behaviour (file splits, renames) go in their own commits, separate from logic
changes, so `git log --follow` and reviews stay readable.

## 5. The `review/` workspace itself
It stays for the duration of the overhaul. Afterwards, ARCHITECTURE.md moves into
`documentation/internals/`, FINDINGS.md becomes the bug log, and `review/` is deleted.
