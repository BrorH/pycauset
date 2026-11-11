# Release Process

This page describes how PyCauset is released to PyPI, and the rules that keep the
PyPI project within its storage quota.

## The rule: only real releases go to PyPI

PyPI caps the **total** size of a project at 10 GB. Every version ever uploaded counts
toward that limit. Yanking a version does not free space; only deleting it does. One
PyCauset release is about **270 MB** (18 wheels across Windows, macOS and Linux for
Python 3.8–3.13, plus the sdist), so the quota holds a few dozen real releases.

In August 2026 the quota filled up because every push to `main` uploaded a new
`0.6.2.devN` build. To make sure that never happens again:

| Action | What happens |
| :-- | :-- |
| Push to `main`, push a tag, open a PR | Nothing is built or uploaded for PyPI. |
| Publish a GitHub **pre-release** | Wheels are built as downloadable artifacts. Nothing is uploaded. |
| Click **Run workflow** on "Publish to PyPI." in the Actions tab | Same: build-only test run, nothing is uploaded. |
| Publish a GitHub **Release** with tag `vX.Y.Z` | Wheels are built and version `X.Y.Z` is uploaded to PyPI. |

The workflow (`.github/workflows/publish.yml`) enforces this with two checks from
`tools/check_pypi_release.py`:

- **Preflight**, before the ~25 minutes of wheel builds: the tag must look like `vX.Y.Z`
  (no `dev`, `rc`, or other suffix), and PyPI must have room for one more release
  below 95% of the quota. It also writes current PyPI usage to the run summary and
  warns above 75%.
- **Verify-dist**, right before upload: every file must be exactly version `X.Y.Z`, and
  the whole upload must be under 450 MB.

## How to release

1. Make sure CI is green on `main` and `CHANGELOG.md` has an entry for the new version.
2. Optional: test the wheel build first. Click **Run workflow** on "Publish to PyPI." in
   the Actions tab, then download the wheels from the run's artifacts.
3. Create the release in one of these ways:
    - **Helper script** (Windows, from the repo root, needs a clean working tree):
      ```powershell
      .\tools\release.ps1              # next patch version, e.g. 0.7.1 -> 0.7.2
      .\tools\release.ps1 -Type minor  # 0.7.1 -> 0.8.0
      .\tools\release.ps1 -SetVersion 1.0.0
      ```
      The script pushes the tag. If the GitHub CLI (`gh`) is installed it also creates the
      GitHub Release; otherwise it prints the link to create it yourself.
    - **By hand:** `git tag v0.7.2 && git push origin v0.7.2`, then on GitHub go to
      *Releases → Draft a new release*, pick the tag, and click **Publish release**.
4. Watch the "Publish to PyPI." run in the Actions tab, then check
   `pip install --upgrade pycauset` in a fresh virtual environment.

Versions come from the git tag through `setuptools_scm`, so there is no version number
to edit in the source. Wheels are built with `cibuildwheel` and uploaded with PyPI
Trusted Publishing (OIDC), so no API token is stored in the repository.

## Checking and freeing PyPI space

See the current usage at any time:

```bash
python tools/check_pypi_release.py preflight
```

To free space, delete old versions you no longer need. Dev builds (`X.Y.Z.devN`) are the
first candidates. On pypi.org, go to *Your projects → pycauset → Manage → Releases*, open a
version, and choose **Delete**. Deleting cannot be undone, and a deleted version number can
never be uploaded again. This can only be done in the web interface, not from CI.

If the project genuinely needs more room, PyPI accepts quota increase requests:
<https://docs.pypi.org/project-management/storage-limits#requesting-a-project-size-limit-increase>.

## Reducing the size of each release

Measured on the v0.6.1 wheels (compressed sizes, as stored on PyPI). One set of three wheels
(Linux, Windows, macOS) for one Python version is 43.7 MB, and 86% of it is the BLAS stack,
not PyCauset:

| Wheel | Total | OpenBLAS | Other bundled libraries | PyCauset itself |
| :-- | --: | --: | :-- | --: |
| Linux x86_64 | 18.9 MB | 11.0 MB (v0.3.15, 2021) | 4.4 MB reference LAPACK/LAPACKE/BLAS, 1.1 MB Fortran/OpenMP runtime | 2.4 MB |
| Windows x64 | 15.0 MB | 12.6 MB | none | 2.4 MB |
| macOS arm64 | 9.8 MB | 7.3 MB | 1.0 MB Fortran/OpenMP runtime | 1.5 MB |

That set is repeated for each of the 6 Python versions (3.8 to 3.13), which gives ~262 MB,
plus the sdist. The wheels contain no stray files, and the Linux binaries are already
stripped of debug symbols.

Options, from cheapest to largest saving:

1. **Trim the sdist** (done): `sdist.exclude` in `pyproject.toml` leaves out docs, images
   and demos. 3.8 MB to 0.5 MB.
2. **Stop shipping a second LAPACK on Linux.** The Linux wheel links LAPACKE from the
   reference netlib packages (`lapack-devel` in `CIBW_BEFORE_ALL_LINUX`) and bundles them
   next to OpenBLAS, which already contains LAPACK and LAPACKE. Saves 4.4 MB per Linux
   wheel, ~26 MB per release. It needs a LAPACKE header that matches OpenBLAS on the
   manylinux image.
3. **Build only supported Python versions.** Python 3.8, 3.9 and 3.10 are all past end of
   life (3.10 ended October 2026). Building 3.11 to 3.13 halves a release to ~135 MB.
   Change `CIBW_SKIP` in `publish.yml` and `requires-python` in `pyproject.toml`.
4. **Stop bundling OpenBLAS.** Depend on `scipy-openblas32`, the OpenBLAS wheel the
   NumPy/SciPy maintainers publish on PyPI for exactly this, and link against it, the way
   NumPy and SciPy do. A PyCauset wheel drops to ~2.5 MB, so a release with 4 Python versions
   is ~30 MB, about 9x smaller than today. It also replaces the 2021 OpenBLAS on Linux with
   the current one on every OS. This is a build-system change: link against the package's
   prefixed symbols, load its library before the extension at import time, and keep the
   wheel-repair tools from re-bundling it. Do a spike first.

With 3 and 4 together, the 10 GB quota holds roughly 300 releases instead of ~37.
