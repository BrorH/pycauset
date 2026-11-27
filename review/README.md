# `review/`: Codebase Review & Health Overhaul workspace

This directory coordinates the full review, cleanup and restructuring of PyCauset
that started on 2026-11-24. It is written for **agents and humans picking up the
work cold**. Read this file first, then the ones it points to.

## Read in this order
0. **`documentation/project/protocols/Adding Operations.md`** and **`Adding Types.md`**: the
   procedures the review established (Phase N). Every change to an operation or type follows them.
1. **`documentation/internals/Architecture.md`**: map of every moving part and how a call
   flows through the stack. Read it before touching code.
2. **[PLAN.md](PLAN.md)**: the phased plan (0–6), what's done, what's next, the
   per-file review checklist.
3. **[FINDINGS.md](FINDINGS.md)**: every concrete issue with ID, severity, repro and status.
4. **[RESTRUCTURE.md](RESTRUCTURE.md)**: proposed moves, deletions and consolidation,
   and the **open decisions** for the maintainer.
5. **[INVENTORY.md](INVENTORY.md)**: every file, with its role and review status.
6. **[TESTS.md](TESTS.md)**: the new compact test suite strategy and migration rules.
7. **[DEADCODE.md](DEADCODE.md)**: dead-code tooling, pipeline, tiered candidates.
8. **[BASELINE.md](BASELINE.md)**: measured numbers and the exact commands to reproduce them.
9. **[LOG.md](LOG.md)**: session-by-session history.

## Directory layout
```
review/
  README.md         this file
  ARCHITECTURE.md   pointer: the map is documentation/internals/Architecture.md
  PLAN.md           phases & checklists
  FINDINGS.md       issue log, F-001...
  RESTRUCTURE.md    cleanup proposal + decisions
  INVENTORY.md      per-file catalog + review status
  TESTS.md          new test-suite design + migration rules
  DEADCODE.md       dead-code program (tools, pipeline, candidates)
  BASELINE.md       numbers + reproduction recipes
  LOG.md            session log
  areas/            per-area deep-review notes (Phase 3), one file per area
  deadcode/         snapshots of dead-code/coverage reports (regenerate per DEADCODE.md §2)
  features/         specs for features accepted into the overhaul (LAZY, VALUE_BOUNDS, ATLAS, BIT_PLANES)
  probes/           runnable investigation scripts (candidates to become tests)
  scripts/          inventory generator, coverage analysers (Python + C++)
```

## Rules for anyone working here
1. **Log every session** in LOG.md: what you did, what you found, what's half-done.
2. **Every issue gets an F-id** in FINDINGS.md before or alongside its fix. Fix
   commits mention the id (`fix(F-001): ...`). Never reuse ids.
3. **Keep `documentation/internals/Architecture.md` true.** If your change moves, adds, deletes or rewires a
   component, update the map in the same commit. Undocumented code is lost code.
4. **Pre-alpha policy:** no backwards-compat shims. Removing something means
   removing its code, tests, docs and plan references entirely. Git is the archive.
5. **Behaviour-preserving moves and logic changes go in separate commits.**
6. **Regression test first** for every bug fix, ideally promoted from a probe.
7. **GPU/CUDA is on hold:** inventory it, don't invest in it.
8. **Disk safety:** never run doc examples or large-N scripts without the size caps
   described in BASELINE.md.
9. **Roles:** the maintainer is creative director (product scope, what PyCauset
   offers); the technical lead decides technical questions and records them in
   RESTRUCTURE §3.1. Ask the maintainer only product questions, in plain language.
10. **R1 priority is the engine** (linear algebra, NumPy parity, storage). Physics
   gets smoke tests and glaring-bug fixes only until after R1.
12. **Guesses may inform, never decide (maintainer, 2026-11-24, clarified).** Results
   (values, dtypes, success/failure) must be deterministic and exact: no sampling or
   heuristic may change them. Guesswork *is* welcome where it can't make a result wrong:
   warnings ("this may overflow"), advice, diagnostics, and explicit statistical
   operations whose documented purpose is to sample (a future `pc.sample_mean`, say).
   If a value can't be guaranteed, accept the larger dtype or raise.
11. **Dead code is a first-class target** (DEADCODE.md). If you touch a file, check
   its never-executed list.

## Current status (update this line every session)
**After session 3 (2026-11-24):** fixed F-001…F-006, F-024, F-025, F-028…F-030, F-033,
F-034, F-035a, F-036…F-040, F-041a; precision_mode removed; lazy expression engine removed
(eager operators, one structure resolver); Arnoldi on Spectra; streaming engine tested.
Session 7: performance (thread-pool defaults, export, BLAS dot: r2_parity mostly at
parity), vectors = 1 x n matrices (F-058), exact integer dot/sum/trace (F-051), linear algebra
on views and complex inputs (F-059..F-063: 5 silent wrong-result classes), user docs GPU-free.
3,860 + 287 tests. Session 6: bit-plane investigation (features/BIT_PLANES.md), exact int
matmul via BLAS (F-055), RAM storage 3× faster (F-057). Session 5: lazy L1/L2 (integer scale
views, C ± Cᵀ closure over bits). Session 4: one class per structure, CUDA to experimental/.
**Next:** PLAN Phase C, the operation atlas (features/ATLAS.md): every operation × dtype ×
structure × view checked against NumPy and classified fast / slow / fallback, generated
report + CI. Then Phase 4 from the atlas report, ASan re-run, docs truth pass (F-008).
Features accepted: lazy evaluation (features/LAZY.md, Phase L), value bounds
(features/VALUE_BOUNDS.md, Phase F1), atlas (features/ATLAS.md, Phase C).

## When the overhaul is finished
(ARCHITECTURE.md has moved into `documentation/internals/`.) FINDINGS.md becomes the
project bug log, and this directory is deleted.
