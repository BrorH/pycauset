# Testing and Bug Tracking Protocol

**Objective:** Maintain a strict protocol for automated testing and bug tracking to keep the project stable and debuggable.

## Bug documentation

Whenever a bug is discovered—whether it's a compilation error, a runtime failure, a logic error, or a regression—it **must** be documented in `tests/BUG_LOG.md`.

### Format

Append a new entry to `tests/BUG_LOG.md` using the following format:

```markdown
## [Date: YYYY-MM-DD HH:MM] Bug Title

**Status**: [Fixed / Open]
**Severity**: [Critical / High / Medium / Low]
**Component**: [e.g., Storage, Matrix Operations, Python Bindings]

**Description**:
A concise description of the issue.

**Reproduction**:
Steps or code snippet to reproduce the failure.

**Root Cause** (if known):
Technical explanation of why it happened.

**Fix** (if applied):
Description of the solution implemented.
```

## Test creation vs. execution

When working on tests, follow this workflow:

1. **Design**: create comprehensive test cases covering edge cases, boundary conditions, and type permutations.
2. **Review**: ensure tests are valid before running.
3. **Execute**: run tests and monitor for failures.

## Regression prevention

When fixing a bug, ensure a regression test is added to the test suite (preferably in a dedicated `test_regressions.py` or the relevant module) to prevent the issue from recurring.
