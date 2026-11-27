# Testing and Bug Tracking Protocol

**Objective:** every bug is written down once, gets a test that fails while it is open, and
the test stays as a regression guard after the fix.

## Bug documentation

Bugs go into `review/FINDINGS.md` as a numbered finding (`F-123`), with:

- **Status:** open, fixed (session / commit), or a question for the maintainer;
- what goes wrong, with the smallest reproduction;
- the root cause, once known, and the fix;
- the test that covers it.

Wrong results of an operation in some dtype, structure or view usually show up first in the
operation atlas (`python -m tests.atlas.report`, `review/atlas/ATLAS.md`); give the cell
that failed in the finding.

## Tests

- A regression test per finding in `tests/engine/test_regressions.py`, named
  `test_F123_<what>`. While the finding is open it is marked
  `@open_bug("F-123")` (a strict xfail): the fix makes it pass, strict-xfail then fails, and
  whoever fixed it removes the marker and updates the finding.
- Value and dtype checks compare against NumPy, the oracle (`tests/engine/test_conformance.py`
  and the atlas); not against values written into the test by hand where NumPy can say.
- The engine suite runs twice in CI, the second time with `PYCAUSET_LAZY=0`.

The suite's layout and commands: `tests/README.md` and `review/TESTS.md`.
