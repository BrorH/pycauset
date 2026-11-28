# NEXT: start here (handoff, 2026-11-28, end of session 9)

You are continuing the PyCauset review and overhaul. This file is the complete handoff: read
it fully before doing anything, then the files it names. Nothing important lives only in a
previous conversation.

## 1. The branch
- All work so far is on branch `ccr-366933fe-71cbew` (331 commits ahead of `main`; `main` is
  an ancestor, so it is a clean continuation). `main` does NOT have this work.
- Start from that branch: `git fetch origin ccr-366933fe-71cbew` and create your own working
  branch from `FETCH_HEAD` (or work on the branch your session is told to use, based on that
  commit). Check `git log --oneline -1` shows "Review notes: thread-pool spin experiment" or
  a later commit from this list. Never start from `main` unless the maintainer has merged it.
- Push only to your session's designated branch. No pull requests unless the maintainer asks.

## 2. Read, in this order
1. `CLAUDE.md` (repository root): build, test, the rules that are easy to break.
2. `review/README.md`: how the review workspace is organised and its rules.
3. `review/PLAN.md`: phases and what is open (`[ ]` and `[~]`).
4. The end of `review/LOG.md` (session 9).
5. `review/RESTRUCTURE.md` §3.1: decisions already made (item 12: session 9). Do not
   re-open them.
6. For any item you pick up: its `review/areas/<area>.md` section and its FINDINGS entry.
7. Before changing an operation or a type: `documentation/project/protocols/Adding
   Operations.md` / `Adding Types.md`, and `documentation/internals/Architecture.md`.

## 3. Setting up a fresh container
```
pip install -e '.[test,dev]' --no-build-isolation   # or: pip install pytest psutil plotly ruff mypy
pip install -r requirements-docs.txt                 # mkdocs for the docs gate
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --target _pycauset -j1       # ~25 min from scratch; -j2 runs out of memory
cp build/cpp/_pycauset*.so build/cpp/libpycauset_core.so python/pycauset/
```
If `pip install -e` tries to build the extension itself, skip it and install the test
dependencies directly; the tests run from `PYTHONPATH=python` against the copied `.so`.

## 4. The gates (all must pass before every commit)
```
PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider                       # ~6,063 passed
PYCAUSET_LAZY=0 PYTHONPATH=python python3 -m pytest tests/engine tests/blockmatrix -q -p no:cacheprovider   # ~6,057
PYTHONPATH=python python3 -m pytest tests/atlas -q -p no:cacheprovider          # 45
(cd build/cpp && ctest)                                                          # 4/4
ruff check --select F python tests benchmarks tools
NO_MKDOCS_2_WARNING=1 mkdocs build --strict -q
```
Counts at the last commit: 6,063 / 6,057 / 45 / 4. A drop in a count is a failure to explain.

## 5. Working rules learned the hard way
- Build with `-j1`. Never edit C++ while a build runs (make keeps the stale object). Never run
  two builds at once. Do not time anything while a build or test run is going.
- Changing a widely included header (e.g. `ParallelUtils.hpp`, `TriangularBitMatrix.hpp`)
  rebuilds nearly everything (~25 min): batch such edits.
- Every fix gets a regression test (`tests/engine/test_regressions.py`, named after the
  review item, e.g. `test_3_8_C4_...`) and a "Fixed (session N)" note in its area file and
  FINDINGS entry. Update `review/LOG.md` at the end of the session; keep PLAN checkboxes true.
- Docs describe the code: change them in the same commit (`documentation/`, BENCHMARKS.md).
- No silent fallbacks, no NumPy behind a PyCauset call, one code path per operation, one
  templated kernel per family (CLAUDE.md). Kernel choice is visible with
  `pc._debug_last_kernel_trace()`; use it in tests.
- Timing: import pycauset BEFORE NumPy (otherwise NumPy's own timings are 10-100x off);
  `pc._native._set_derived_cache(False)` to time computations, not cached answers.
  `perf` does not work in this container. `ParallelFor` runs ranges under 32 items inline;
  a pool dispatch costs ~25 us here.
- Commit with a message saying what changed and why (review item / F-id), then push. Use the
  commit trailer your session's instructions give you.

## 6. State at handoff
- Phase 3 (area reviews): every R1 area is reviewed; the status of each is in the PLAN.md
  Phase 3 table (updated at this handoff). Leftovers worth doing next are in §7; the
  area files' "Open:" lines are the full list.
- Phase 6 (speed): PLAN 6.5 (causal-matrix kernels) done tonight: `C @ C` at n = 2000 6 ms
  vs NumPy float32 25 ms; `(C @ C) @ C` 24 vs 72 ms; bit matvec 0.007 ms at n = 256.
  `benchmarks/parity.py` (2026-11-28): 8 of 12 rows at parity; below: dot 0.39x, import
  0.83x, save 0.84x, matmul 0.90-1.2x (noisy).
- Tried and reverted (do not retry as is): 2,048-element dot blocks; a spinning thread pool
  (PLAN 6.2 records why).

## 7. Work queue, in this order
1. **3.8 C-4, the rest:** structured constructors and `causal_matrix(data)` validated and
   packed in the engine, `diagonal(1-D)` without an n x n array, `pack_as` row-wise. The
   full design (engine functions, bindings, Python) is written in
   `review/areas/3.8-python-facade.md` under "Plan for the rest". Regression tests: no NumPy
   export for PyCauset input (`export_count` unchanged), `diagonal(np.arange(4000.))` peak
   below 1 MiB, `causal_matrix(1500 x 1500)` fast, the same warnings as today.
2. **3.6 C-16, the rest** (`review/areas/3.6-math-entry.md`): one name for the four
   normaliser helpers (`materialize`, `write_out`, `cast_to`, `to_dense`),
   `BitClosureMatrix::value` (a private copy of the view rule: use `ViewValue.hpp`), the
   ignored `result_file` parameters (delete them).
3. **3.4 C-16, the rest** (`review/areas/3.4-dispatch-kernels.md`), then 3.4's other "Open:"
   lines (GIL for element-wise ops, BLAS dimension checks for gemv / gemm, include blocks).
4. **3.2 C-10:** the antisymmetric `-A` diagonal sign (`review/areas/3.2-object-model.md`).
5. **3.8 C-9:** generate the stub's class sections from the bindings, plus a mypy check in
   the tests.
6. **PLAN 1.8:** warnings-as-errors build for our own code.
7. **Phase 6:** 6.6 (import / save rows), 6.0 (one OpenBLAS per process: the real fix for the
   dot gap), 6.2 (thread pool), 6.4 (out-of-core benchmark on a real disk).
8. Then the remaining PLAN phases (4.x cleanup, L lazy features, F1 value bounds, D derived
   cache sections, 5.2 API docs) in PLAN order.

## 8. For the maintainer, not the agent
- Windows run: `pytest tests/engine tests/blockmatrix`, `python benchmarks/parity.py`. Check
  that MSVC compiles (`__int128` is used in the integer kernels and will likely need
  replacing), FILE_SHARE_DELETE, the GIL release, the generational block saves, and the AVX2
  loops (MSVC takes the plain-loop path).
- `scripts/fakecg.sh` (a faked RAM budget for streaming tests) needs `unshare -m`.
