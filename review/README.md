# `review/`: Codebase Review & Health Overhaul workspace

This directory coordinates the full review, cleanup and restructuring of PyCauset
that started on 2026-11-24. It is written for **agents and humans picking up the
work cold**. Read this file first, then the ones it points to.

## Read in this order
1. **[ARCHITECTURE.md](ARCHITECTURE.md)**: map of every moving part and how a call
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
  ARCHITECTURE.md   the map (keep it true!)
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
  features/         specs for new features accepted into the overhaul (VALUE_BOUNDS.md, LAZY.md)
  probes/           runnable investigation scripts (candidates to become tests)
  scripts/          inventory generator, coverage analysers (Python + C++)
```

## Rules for anyone working here
1. **Log every session** in LOG.md: what you did, what you found, what's half-done.
2. **Every issue gets an F-id** in FINDINGS.md before or alongside its fix. Fix
   commits mention the id (`fix(F-001): ...`). Never reuse ids.
3. **Keep ARCHITECTURE.md true.** If your change moves, adds, deletes or rewires a
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
Session 4: CUDA to experimental/, Structure enum, one class per structure (F-032/F-045/F-046
fixed), lazy-evaluation design (features/LAZY.md). 2,825 passed / 33 xfailed in ~24 s; ASan+UBSan clean (sanitizer builds
need -j1). **Open priorities:** F-041b (maintainer: zeros() default dtype); F-032 + renames
(PLAN 4.6); F-040 follow-ups (views in the tiler, `set_memory_threshold` vs streaming);
performance F-012 (profile the 13× element-wise gap); dead-code Tier B; docs F-008. All decisions answered (RESTRUCTURE §3); new feature
accepted: value bounds (features/VALUE_BOUNDS.md, PLAN Phase F1); lazy evaluation
(features/LAZY.md, PLAN Phase L).

## When the overhaul is finished
ARCHITECTURE.md moves into `documentation/internals/`, FINDINGS.md becomes the
project bug log, and this directory is deleted.
