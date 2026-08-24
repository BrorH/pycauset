# Third-Party Notices

PyCauset builds on and bundles the following open-source components. Each is
distributed under its own license; this file records the attribution required
by those licenses. For PyCauset's own license, see `LICENSE` (MIT).

## Bundled / vendored (compiled into or shipped with the package)

### Eigen (3.4.0)
- **License:** Mozilla Public License 2.0 (primary). Some files contain
  third-party code under BSD-2/3-Clause, LGPL 2.1, or MINPACK licenses.
- **Attribution:** © Eigen authors, <https://eigen.tuxfamily.org>
- **Note:** the build may define `EIGEN_MPL2_ONLY` to restrict usage to the
  MPL2/BSD-permissive subset.
- Source: `include/eigen3/` (vendored); also fetched via CMake FetchContent.

### OpenBLAS (0.3.26)
- **License:** BSD-3-Clause (with some permissive components under other
  BSD-style licenses).
- **Attribution:** © OpenBLAS contributors, <https://www.openblas.net/>
- Source: binaries fetched via CMake FetchContent
  (`OpenBLAS-0.3.26-x64.zip`); shipped as `libopenblas.dll` / `.so` / `.dylib`.

### pybind11 (2.12.0)
- **License:** BSD-3-Clause.
- **Attribution:** © 2016 Wenzel Jakob <wenzel.jakob@epfl.ch> and contributors.
- Source: fetched via CMake FetchContent (v2.12.0).

### GoogleTest (1.14.0) — test-only
- **License:** BSD-3-Clause.
- **Attribution:** © Google Inc. and contributors.
- Not shipped in the runtime package; used only by the C++ test targets.

## Build-time dependencies (not bundled in the runtime package)

- **scikit-build-core** — Apache-2.0 / BSD-style (see project metadata).
- **setuptools_scm** — MIT.

## Optional accelerator

- **CUDA Toolkit** (optional, `ENABLE_CUDA`): NVIDIA CUDA is proprietary and is
  **not bundled**. GPU support requires a user-installed CUDA Toolkit; see
  NVIDIA's EULA at <https://docs.nvidia.com/cuda/eula/>.
