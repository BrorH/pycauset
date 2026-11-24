# Work Log

One entry per working session, newest at the bottom. Say what you did, what you
found, and what you left half-done, so the next agent can continue without
re-deriving anything. Keep entries factual and short. Details belong in
FINDINGS / ARCHITECTURE / area notes; link to them.

---

## Session 1: 2026-11-24, kickoff (Claude Code, branch `ccr-366933fe-71cbew`)

**Brief from the maintainer:** start a thorough review and health check of the
whole repository: catalog everything, plan the multi-week overhaul, document it in a
dedicated directory for future agents, test heavily, and propose cleanup and
restructuring. Pre-alpha, so no backwards compatibility. GPU work is on hold. Mid-session the
maintainer asked for an overview and plan *before* deep testing, and for more
verbose progress reporting. The documents below were written after that steer.

**Done**
1. Built from source (pip/scikit-build, plus standalone CMake with `BUILD_TESTS`,
   plus an ASan+UBSan build). Environment gotchas recorded in BASELINE.md.
2. Baseline: Python suite 799 passed / 34 skipped. 6/6 wired gtests pass. Lint and
   warning counts recorded.
3. Un-skipped the three "crash/segfault/unverified" tests: all pass (F-009).
4. Built the three orphaned gtests: `test_parallel_utils` hangs → **F-004** (nested
   ParallelFor deadlock). Reading the code also gave F-005.
5. Wrote four probes (`review/probes/`):
   - NumPy-fallback detector → eigh/eigvalsh/svdvals/matrix_rank always use NumPy
     (F-011); `cond` uses the wrong default norm (F-003).
   - Out-of-core consistency → core ops correct when forced to disk.
   - dtype × op grid → **F-001** (int32 `A*3` is a silent no-op), F-002 (no `-A`,
     `3*A` fails, ...), F-013 (bool semantics).
   - Doc-snippet runner → F-008 (≈20 broken examples).
6. ASan+UBSan run (first attempt, UB fatal): stopped at the first UB, signed overflow
   in the int32 add kernel → **F-006**. A second build with UB set to recoverable was
   started to get a full pass. See "Left open".
7. Stray `_vals`/`_vecs` files in the repo root led to **F-024**: every `eig`
   result is backed by the same `./_vals` file, so a second `pc.eig()` silently
   corrupts the first. It was masked by `.gitignore` entries.
8. Read the architecture end to end and wrote ARCHITECTURE.md. Wrote INVENTORY.md
   (154 source files with a review column), FINDINGS.md (F-001…F-024), PLAN.md
   (phases 0–6), RESTRUCTURE.md (proposal plus 8 open decisions), BASELINE.md.

**Incidents**
- The first uncapped doc-snippet run filled the disk (a doc example allocates a
  ~125 GB causal set; see F-008). That killed the parallel ASan build. Fixed by
  capping file size in the runner and deleting the working dirs.

**Left open / next steps**
- Maintainer to answer RESTRUCTURE.md §3 decisions.
- Finish the full ASan+UBSan pass and log the results (BASELINE row, new F-ids).
- Start PLAN Phase 1 (safety net). Suggested first PRs: 1.1 test harness/conftest,
  1.2 wire gtests into CI, 1.4 remove stale skips.
- Not yet done: coverage measurement (1.9); checking whether any production
  code path actually nests ParallelFor (affects F-004 severity).

## Session 1, part 2: 2026-11-24, new direction from the maintainer

**Steer:** (1) design a new, compact, smarter test suite instead of keeping the
16.7k-line one; (2) the maintainer is creative director and the agent is technical
lead, so make technical calls and ask only plain-language product questions;
(3) make dead-code elimination a first-class part of the overhaul, and look for
existing tools; (4) R1 = the engine; physics is deprioritized.

**Done**
1. Full ASan+UBSan pass: no memory errors; UB in int32 add/mul (F-006) and a null
   memcpy (F-025). Learned that pytest capture hides UBSan output (F-027).
2. Dead-code tooling researched and trialled: Skylos (best static Python), Vulture
   (noisy), cppcheck (useful only intersected with coverage), coverage.py with
   per-test contexts, and gcov/gcovr. Wrote `scripts/coverage_analysis.py` and
   `scripts/cpp_unexecuted.py`; snapshots in `review/deadcode/`. Wrote DEADCODE.md.
3. Coverage results: Python 76% (87 never-executed functions); C++ 35% lines /
   489 of 1,562 functions never executed; **StreamingManager.cpp 0%** (F-026); a
   pure-Python fallback matrix still wired into `pc.matrix` but never reached.
4. Test redundancy: 47/121 test files contribute zero unique Python lines. Wrote
   TESTS.md (oracle-first, table-driven, invariant-based; ~3–4k lines target;
   migrate, then retire old files only with coverage proof).
5. RESTRUCTURE §3 rewritten: 9 technical decisions made (vetoable), 2 product
   questions left open. PLAN updated: R1 engine priority, Phase 1 = new suite core,
   new Phase 1b = Tier-A dead code, physics review moved post-R1.

**Next:** Phase 1 (new-suite core, CI wiring) and Phase 1b (Tier-A removals), in
parallel. The C++ coverage run should be repeated with the gtests included before
removing MemoryGovernor's pinning API.
