# Test-Suite Redesign

**Mandate (maintainer, 2026-11-24):** replace the current ~16.7k-line suite with a
new, smarter, more compact one. Old tests may be outdated or may not reflect what
we're trying to do. Test runs cost CPU **and agent tokens**.

**Status:** strategy written; measurements in §2; nothing migrated yet.

---

## 1. What's wrong with the current suite (measured)

| Problem | Evidence |
|---|---|
| Green while serious bugs exist | F-001 (int scalar multiply no-op), F-024 (eig results corrupt each other), F-002 (no `-A`) all pass the suite |
| Heavy overlap | 47 of 121 test files cover **zero** Python lines that no other file covers (`review/scripts/coverage_analysis.py tests`). C++-side uniqueness still to be measured, see §2 |
| Organized by *when*, not *what* | `*_r2`, `*_phase_b…h`, `test_r1_*`, `phase3_4_*` |
| Untested public API | `pc.divide` is never called; the dtype-less `zeros()` wrapper's 18 operators never run |
| Stale skips | 3 "crashing" tests pass (F-009) |
| Fragile harness | 15 files hack `sys.path`; CI copies `.so` files into the source tree (F-023) |
| C++ barely tested | 9 gtest files, 3 never built, none run in CI |
| Hidden sanitizer output | pytest capture swallows UBSan reports unless `-s` is passed |

Runtime is **not** the problem: 21 s for 799 tests. The costs are false
confidence, maintenance and context size (16.7k lines that agents must read), and
verbose output.

## 2. Measurements that drive the migration
- Python per-test-file coverage: done (2026-11-24). 47/121 files have 0 unique lines.
- C++ line coverage of the whole suite, and per test file: **pending** (`build/cov`).
- A test file is *redundant* only if it has 0 unique lines in **both** Python and C++.

## 3. Design of the new suite

### 3.1 Principles
1. **Oracle-first.** NumPy is the specification for the engine. Most tests are a
   generic comparison against NumPy, not hand-written expected values.
2. **Tables, not files.** Supported dtypes, structures and ops live in one spec
   table. Tests are generated from it. Adding a dtype means adding a row, not a file.
3. **Invariants over examples.** Check properties that must always hold (element
   access == export == sum == matmul-with-identity, for every view: scaled,
   transposed, conjugated, sliced, spilled to disk).
4. **Every bug gets one tiny regression test**, named after its F-id.
5. **Quiet by default.** One summary line on success; failures print only the short
   traceback. The default tier finishes in < 30 s on 4 cores.
6. **No path hacks:** tests run against the installed (editable) package.

### 3.2 Layout
```
tests/
  conftest.py            fixtures: tmp backing dir, memory threshold, force_disk,
                         numpy-oracle helpers, dtype/structure catalog
  spec.py                THE support table: ops × dtypes × structures → expected
                         behaviour (match NumPy | documented divergence | raises X)
  engine/
    test_conformance.py  generated grid: every op in spec.py vs NumPy, values + dtype,
                         RAM and forced-disk, square/rect/1×1/0×n shapes
    test_invariants.py   view-metadata consistency, CoW isolation, result
                         independence (catches F-001 and F-024 classes of bugs)
    test_linalg.py       identities (inv(A)@A≈I, solve, det(AB)=det A det B, QR/LU/SVD
                         reconstruct, eig residuals), optionally via Hypothesis
    test_persistence.py  save/load round-trip for every class in the catalog;
                         crash-consistency (A/B header slots); container corruption
    test_storage.py      governor spill/promote, backing dir, temp-file lifecycle,
                         export guard, out-of-core equals in-RAM
    test_interop.py      NumPy in/out, buffer protocol, ufunc policy
    test_api_surface.py  every name in __all__ resolves and has a doc page; .pyi in sync
    test_regressions.py  one test per fixed F-id
  physics/               SMOKE ONLY for R1: sprinkle → validate → dimension estimate,
                         field propagator shape/symmetry, plots render. (Physics is
                         deprioritized for R1.)
  docs/test_snippets.py  every python block in user docs runs (size-capped)
  cpp/                   gtest: ParallelUtils, MemoryGovernor, MemoryMapper, CoW,
                         container header; all wired into CMake and CI
```
Markers: `slow` (large N, stress, threading; off by default), `gpu` (on hold).

### 3.3 Size target
Roughly 3–4k lines of test code (vs 16.7k), generating *more* cases than today.
The conformance grid alone should reach ~5–10k parametrized cases at the cost of a
few hundred lines.

## 4. Migration rules (how we retire the old suite safely)
1. Build the new suite in `tests/` next to the old one (old stays in `tests/python/`
   until retired).
2. New spec rows for **known-open bugs** are marked `xfail(strict=True, reason="F-0xx")`,
   so the fix flips them green and the marker must be removed.
3. Before deleting any old file, check that every Python and C++ line it uniquely
   covers is either covered by the new suite or confirmed dead (and the code
   deleted). Use `coverage_analysis.py` plus the C++ coverage report.
4. Delete old files in batches, one subject at a time (see the INVENTORY test groups),
   and record each batch in LOG.md.
5. When `tests/python/` is empty: remove the CI `.so` copy step and the `sys.path`
   hacks, and make sanitizer jobs run with `-s`.

## 5. Running tests cheaply (for agents)
- Default: `pytest -q -p no:warnings --tb=short` → one line on success.
- While working on one area: `pytest tests/engine/test_conformance.py -k matmul`.
- Never paste full verbose logs into context; grep for `FAILED|Error`.
- Slow tier and sanitizers run in CI, not in every agent loop.
