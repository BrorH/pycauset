# macOS wheel portability (R2_HARDEN, remaining)

## Current state

`publish.yml` pins `MACOSX_DEPLOYMENT_TARGET="15.0"` and installs `openblas` and
`libomp` from Homebrew. Homebrew bottles are built against the runner's macOS SDK,
so the produced wheels require **macOS 15 or newer**. This is an honest, working
state (the explicit `15.0` target prevents a wheel that claims macOS 12 but links
libraries that need 15).

## Goal

Wheels that install and run on older macOS (e.g. 12 or 13), matching the project's
other portability claims.

## Why this is deferred, not done

Both native dependencies are the constraint, and each has a tradeoff:

1. **OpenBLAS** (needed for `lapacke.h`/BLAS; Apple Accelerate does not ship
   `lapacke.h`, so it cannot replace OpenBLAS). Building it from source needs a
   Fortran compiler (`gfortran`) and OpenBLAS's Makefile must honor the deployment
   target, which is not automatic. The CMake `FetchContent` fallback in
   `CMakeLists.txt` already knows how to build OpenBLAS from source, so the hook
   exists; the work is wiring `gfortran` + the deployment target through
   `CIBW_BEFORE_ALL_MACOS` / `CIBW_ENVIRONMENT_MACOS`.
2. **libomp** (OpenMP runtime for the `#pragma omp` loops). Dropping it makes the
   OpenMP loops run serially (correct, but slower); building it from source is
   C/C++ and simpler than OpenBLAS but still non-trivial in cibuildwheel.

The verification problem is the harder part: the wheel must be *installed and run*
on an older macOS to prove the claim. GitHub offers `macos-13` (Intel) and
`macos-14`/`macos-15` (arm64), so an arm64 wheel built with a lower target could be
smoke-tested on `macos-14`, but a genuine macOS 12 claim needs a separate runner.

## Plan

1. `CIBW_BEFORE_ALL_MACOS`: install `gfortran` (for OpenBLAS) and keep or build
   `libomp`; drop `brew install openblas` so CMake takes the `FetchContent`
   OpenBLAS path.
2. `CIBW_ENVIRONMENT_MACOS`: lower `MACOSX_DEPLOYMENT_TARGET` and thread it into
   the OpenBLAS build (via `CMAKE_C_FLAGS`/`CMAKE_Fortran_FLAGS` or OpenBLAS's own
   `MACOSX_DEPLOYMENT_TARGET`).
3. Add a wheel smoke-test job on `macos-14` (arm64) that installs the built wheel
   and imports `pycauset`, proving the lower deployment target actually works.
4. Keep the explicit deployment target honest: never claim a target lower than what
   the bundled libraries actually support.

## Decision

Deferred: the current macOS 15+ wheels are correct and honest, and the from-source
build is a real, multi-step packaging change with a hard verification requirement.
It is intentionally left documented here rather than half-done.
