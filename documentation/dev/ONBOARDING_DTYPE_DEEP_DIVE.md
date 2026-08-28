# Onboarding: PyCauset dtype deep-dive

This document is a self-contained handoff for a fresh agent instance. Read it fully
before touching anything.

## 1. What PyCauset is

PyCauset is "NumPy for causal sets": a high-performance Python library for Causal Set
Theory, backed by a C++ engine (pybind11 + Eigen + OpenBLAS, optional CUDA). It exposes
matrices/vectors with a NumPy-like API, memory-mapped / out-of-core storage, and a
"properties-as-gospel" semantic structure system.

Repo: `C:\Users\ireal\Documents\Projects\pycauset`

Release-1 (R1) gate (authoritative): ship on **correctness + no silent wrong answers +
explicit support status everywhere**. Optimization (>= 0.90x NumPy, GPU parity,
streaming-everything) is deferred to a continuous post-R1 program. R1 is nearly shipped;
the remaining work is release mechanics (CI green, tag v0.5.1, PyPI publish).

## 2. Non-negotiable rules (from the project owner)

- **"THE DOCS MUST STAY UP TO DATE. No exceptions."** Every code change ships with its
  test AND its doc line, in the same commit.
- **NO em-dashes ever** (never type `—` or `–` in any prose/docs you write; use commas,
  colons, or hyphens).
- **No silent wrong answers.** Anything unsupported must be an explicit error or a
  documented support status, never a silently-wrong result.
- **"Deprecation = Purge."** Remove dead/deprecated things outright; never leave "this is
  deprecated" placeholders.
- **"PyCauset is supposed to WORK LIKE NUMPY."** API naming, dtype tokens, and defaults
  should match NumPy expectations wherever possible.
- The user is the "creative director"; you are the "technical executor". Work slowly and
  explain what you changed. Assume real money and professional standards.

## 3. Environment / how to build and test

- **Always use the venv Python**: `.venv\Scripts\python.exe`. It already resolves
  `import pycauset` to the repo's `python/pycauset/`. The system `python` resolves to a
  STALE site-packages copy (missing `determinant`, `matrix_rank`, etc.) - do not use it.
  Alternatively set `$env:PYTHONPATH="C:\Users\ireal\Documents\Projects\pycauset\python"`.
- **Run tests**: `.venv\Scripts\python.exe -m pytest tests/python -q`. On Windows/MSVC the
  baseline is ~569 passed / 29 skipped / 0 failed. Don't break it.
- **Rebuild the C++ extension** (only needed when you edit `src/` or `include/`): the
  canonical build is `build_msvc` (Visual Studio 18 2026, MSVC, Release, CPU-only).
  Deploy = copy `build_msvc\Release\_pycauset.pyd`, `pycauset_core.dll`, and
  `libopenblas.dll` into `python\pycauset\`.
- **`test.py` at the repo root is the USER's scratch file.** Read it to see what they are
  trying (it currently exercises the dtype pain points below). Do NOT modify it.

## 4. Architecture map (where everything lives)

Python (public + internal facade):

- `python/pycauset/__init__.py` - the facade. Public functions (zeros/ones/empty/dot/
  matmul/...), `_extra_exports` list, and `__all__` (which was recently curated to
  exclude internal native machinery). The dtype tokens are module-level constants here.
- `python/pycauset/_internal/ops.py` - most linalg ops implemented via composition +
  NumPy fallback. `matmul` (dispatch + dense fallback), `_effective_structure_for`
  (structure detection), `_try_convert_to_dense_f64` etc.
- `python/pycauset/_internal/dtypes.py` - `normalize_dtype` (the canonical dtype-token
  normalizer). This is where dtype aliases like `bool` vs `bool_` belong.
- `python/pycauset/_internal/matrix_api.py` - the `Matrix` factory shim (`__new__`
  dispatches on dtype to the native class).
- `python/pycauset/_internal/properties.py` - the properties-as-gospel system
  (`obj.properties["is_symmetric"]=True`, `effective_structure_from_properties`).
- `python/pycauset/_internal/persistence.py` - save/load.
- `python/pycauset/_internal/export_guard.py` - NumPy export dtype inference + guards.
- `python/pycauset/_internal/patching.py` - patches native classes (`.properties`,
  `__init__` wrapper, etc.).

C++ (native engine):

- `src/bindings/` - pybind11 bindings (bind_matrix.cpp is huge; the "unsupported matrix
  multiplication types" and dtype-warning strings live in/around here and Promotion).
- `src/core/PromotionResolver.cpp` + `include/pycauset/core/PromotionResolver.hpp` - the
  C++ dtype-promotion rules (this is where bool/int matmul promotion lives).
- `src/compute/cpu/CpuSolver.cpp` - the compute kernels + LAPACK paths.
- `src/matrix/`, `src/vector/`, `src/core/` - matrix/vector types, MemoryMapper,
  MemoryGovernor, StorageUtils, ObjectFactory.

Structure system: `_internal/ops.py::_effective_structure_for(obj)` returns one of
"zero" / "identity" / "diagonal" / "symmetric" / "antisymmetric" / "upper_triangular" /
"lower_triangular" / "general". Native structural types (IdentityMatrix, DiagonalMatrix,
SymmetricMatrix, AntiSymmetricMatrix) are recognized by type name.

## 5. The dtype situation (current state + the pain points to fix)

Current dtype tokens exposed on `pycauset`: `bit`, `bool_`, `int8`, `int16`, `int32`,
`int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`,
`complex_float16`, `complex_float32`, `complex_float64`.

Known problems the user is angry about (all reproduced below):

1. **`zeros` / `ones` / `empty` require a `dtype=` keyword.** Their signatures are
   `def ones(shape, *, dtype, **kwargs)` (dtype is required keyword-only). NumPy's
   `np.ones(shape)` defaults to float64. The user wants these to "just work" with no
   dtype, starting from the smallest sensible dtype and promoting upward as needed.
   Specifying a dtype must remain optional.

2. **`pc.bool` does not exist; only `pc.bool_`.** NumPy uses `np.bool_` for the dtype but
   `np.dtype('bool')` also works, and users reach for `pc.bool` first. Add a `bool` alias
   (and make dtype normalization accept it).

3. **Bool/bit matrix multiplication throws.** `B @ C` where `B` is a bool matrix and `C`
   is the causal matrix (`TriangularBitMatrix`) fails with "unsupported matrix
   multiplication types" (and/or a "mixing type error" warning). Bool matmul must
   promote (like NumPy's bool arrays promote in np.dot) and just work.

4. **`pc.dot(B, C)` does not work** for the user's inputs (matrices). Investigate the dot
   dispatch: it is currently vector-oriented (`def dot(a, b) -> float | complex`). Decide
   whether `dot` should accept matrices (NumPy's `np.dot` does both matmul and vector
   dot).

5. **The `@` operator for mixed/structured types is too strict.** It should degrade to a
   correct dense/NumPy path rather than throwing, when a specialized native kernel does
   not exist.

## 6. Exact reproductions (verify these fail, then make them pass)

Create a scratch script (not `test.py`) with, roughly:

```python
import pycauset as pc
N = 8
C = pc.CausalSet(N, seed=1223).C          # TriangularBitMatrix
B = pc.ones((N, N), dtype=pc.bool_)       # currently requires dtype; should not

# these should all work after the fix:
print(pc.ones((N, N)))                    # default dtype, no keyword
print(pc.zeros((2, 2)))                   # default dtype
print(pc.empty((2, 2)))                   # default dtype
print(pc.bool)                            # alias for pc.bool_ (should resolve)
print(B @ C)                              # bool @ bit matmul (promote)
print(pc.dot(B, C))                       # dot should accept matrices
```

Run it with `.venv\Scripts\python.exe`. Each line that currently raises is a fix target.

## 7. Your mission

Do a careful deep-dive on dtype handling, in this order, and fix each with a test + doc
line in the same commit:

1. Read `test.py`, `python/pycauset/__init__.py` (zeros/ones/empty/dot/matmul + the dtype
   constants), `python/pycauset/_internal/dtypes.py` (`normalize_dtype`), and
   `src/core/PromotionResolver.cpp` to understand the current promotion rules end to end.
2. Give `zeros`/`ones`/`empty` a sensible default dtype (float64, matching NumPy) so the
   `dtype=` keyword becomes optional. Make sure explicit dtype still works.
3. Add `bool` as an alias for `bool_` in `normalize_dtype` (and expose `pc.bool`), while
   keeping `bool_` working for back-compat.
4. Fix bool/bit matmul (`@`) so `B @ C` promotes correctly instead of throwing. The error
   strings ("unsupported matrix multiplication types", "mixing type error") come from the
   C++ promotion/dispatch path - find them and make the matmul dispatch degrade to a
   correct dense/NumPy path when no specialized kernel exists.
5. Make `pc.dot` accept matrices (and do the right thing for vector-vector / matrix-vector
   / matrix-matrix), matching NumPy's `np.dot` semantics.
6. Keep the properties-as-gospel structure shortcuts intact (identity/zero/diagonal/
   symmetric/antisymmetric/triangular) - do not regress them.
7. Run the full suite after each change; keep it green on Windows (and note anything that
   looks Linux/macOS-specific).

## 8. Recent context you should know

- A long CI-unblocking pass just happened: Linux `-mbig-obj` (MinGW flag leak), LAPACKE
  header installs, macOS OpenMP imported-target, macOS `:memory:` fd bug (POSIX
  `shm_open`, recently fixed with a short name `/pc<pid>x<counter>`), and a threaded-test
  segfault (concurrent native construction is not thread-safe; the test now serializes
  construction). These are in flight on GitHub Actions; don't be surprised by CI noise.
- `pc.det` and `pc.rank` were just added as aliases for `pc.determinant` and
  `pc.matrix_rank`.
- `__all__` was curated: internal native machinery (LazyMatrix, lazy_*, MemoryGovernor,
  OpContract/OpRegistry, get/set_storage_root, make_coordinates, sprinkle) is excluded.

## 9. Files you must NOT touch

- `LICENSE` (owner's year change), `tests/python/test_factorizations.py` (owner's skip).
- `README.md`, `mkdocs.yml`, `overrides/main.html`, `documentation/project/Philosophy.md`,
  and the logo assets - these are another agent's in-flight website/docs work.
- `benchmarks/benchmark_*.py` and other untracked scratch under `benchmarks/` (gitignored).
- `scripts/`, `test.py` (owner's scratch), and the docs-agent's `documentation/project/plans/`
  and `mockups/`.

## 10. Definition of done for this mission

- `pc.ones/zeros/empty` work with no dtype argument (default float64) and still accept an
  explicit dtype.
- `pc.bool` resolves (alias of `pc.bool_`), and dtype normalization accepts `bool`.
- Bool/bit `@` matmul works and produces a correct promoted result (no "unsupported
  matrix multiplication types", no "mixing type error").
- `pc.dot` accepts matrices and matches NumPy `np.dot` semantics.
- Each fix has a test (in `tests/python/`) and a doc/CHANGELOG line in the same commit.
- Full suite green on Windows MSVC. No em-dashes anywhere.
