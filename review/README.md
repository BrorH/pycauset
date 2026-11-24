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
6. **[BASELINE.md](BASELINE.md)**: measured numbers and the exact commands to reproduce them.
7. **[LOG.md](LOG.md)**: session-by-session history.

## Directory layout
```
review/
  README.md         this file
  ARCHITECTURE.md   the map (keep it true!)
  PLAN.md           phases & checklists
  FINDINGS.md       issue log, F-001...
  RESTRUCTURE.md    cleanup proposal + decisions
  INVENTORY.md      per-file catalog + review status
  BASELINE.md       numbers + reproduction recipes
  LOG.md            session log
  areas/            per-area deep-review notes (Phase 3), one file per area
  probes/           runnable investigation scripts (candidates to become tests)
  scripts/          helpers for maintaining these docs (inventory generator)
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
9. Don't decide RESTRUCTURE "DECISION" items yourself; they belong to the maintainer.
   Record answers inline once given.

## Current status (update this line every session)
**2026-11-24:** Phase 0 (orientation & baseline) complete except a full sanitizer
pass. Waiting on the maintainer's RESTRUCTURE decisions; Phase 1 (safety net) can
start now. Critical open bugs: F-024 (eig results corrupt each other), F-001 (int scalar
multiply is a no-op), F-004 (nested ParallelFor deadlock), F-006 (UB).

## When the overhaul is finished
ARCHITECTURE.md moves into `documentation/internals/`, FINDINGS.md becomes the
project bug log, and this directory is deleted.
