# Release Process

How a version reaches PyPI.

## Versions

The version comes from the latest git tag (`vX.Y.Z`, read by `setuptools_scm` through
scikit-build-core). Untagged commits build as `X.Y.(Z+1).devN`.

## Steps

1. **Tag.** `./tools/release.ps1` (Windows; `-Type patch|minor|major`, default patch, or
   `-SetVersion 0.7.0`) reads the latest tag, asks for confirmation, then creates and pushes
   the next one. Without PowerShell: `git tag v0.7.0 && git push origin v0.7.0`.
2. **Publish a GitHub Release** for that tag (the Releases page, or `gh release create
   v0.7.0`). Pushing a tag alone publishes nothing.
3. **The workflow** `.github/workflows/publish.yml` runs on the published release (or by
   hand, `workflow_dispatch`): `cibuildwheel` builds 64-bit wheels for Linux, Windows and
   macOS for every Python the package supports (`requires-python`), plus the sdist, and
   uploads them to PyPI with Trusted Publishing (OIDC).

## Before tagging

- The gates are green on CI (`ci.yml`: the suites on Linux, Windows and macOS, the C++
  tests), and the operation atlas has no wrong cells.
- `CHANGELOG.md`'s Unreleased section becomes the version's section.
