# Repository Hygiene (Source vs Artifacts)

## Rule: do not commit compiled artifacts

PyCauset is a compiled extension project. The repository should contain:
- source code,
- documentation,
- tests,
- build configuration.

It should **not** contain compiled build outputs such as:
- `_pycauset.pyd` / `_pycauset.so`,
- `.dll` / `.so` / `.dylib` runtime libraries,
- CMake build directories.

## Why

Committing binaries causes:
- **stale engine confusion** (Python imports an old binary while you edit new C++ code),
- noisy diffs,
- installation/version mismatch issues.

Since users install via `pip install pycauset` (wheels), binaries belong in **release artifacts**, not in git.

## Enforcement

- Add `.gitignore` rules to exclude these artifacts.
- Purge previously committed artifacts from git history (single-maintainer repo).

## History purge (planned)

We will rewrite history to remove committed artifacts.

Preferred tool: `git filter-repo`.

High-level steps (to be executed deliberately):
1. Remove tracked binaries from the current tree.
2. Run filter-repo to excise them from history.
3. Force-push the cleaned history.
4. Re-clone locally.

Because this repo is single-maintainer, the usual collaboration risks are minimal.
