# Baseline: build, test, lint, performance

Measured numbers at the start of the overhaul, so later work can be compared
against them, plus the exact recipes to reproduce them. Update the numbers (with the
date) whenever a phase lands, and keep the old row so the trend is visible.

## Environment of the first baseline
- 2026-11-24, commit `046c5ab`, Linux x86-64 (Ubuntu 24.04 container), 4 vCPU, 15 GB RAM
- Python 3.13, NumPy 2.5.3, GCC 13.3, CMake 3.28, Ninja, OpenBLAS 0.3.26 (apt)

## Results

| Metric | 2026-11-24 |
|---|---|
| Python tests (`pytest tests/python`) | **799 passed, 34 skipped**, 21 s |
| … of the skips: GPU-only | 27 |
| … of the skips: stale (pass when un-skipped) | 3 (F-009) |
| C++ gtests wired in CMake | 6/6 pass |
| C++ gtests orphaned | 3 (2 pass, 1 hangs, F-004) |
| ASan+UBSan Python suite (with `-s`) | **no memory errors**; 3 UB sites: int32 add & mul overflow (F-006), null memcpy (F-025) |
| Python line coverage (suite) | 76%; 87 functions never executed |
| C++ line / function coverage (suite + demos, no gtests) | 35% / 45%; 489 of 1,562 source functions never executed; StreamingManager 0% |
| Test files with 0 unique Python lines | 47 of 121 |
| Compiler warnings (`-Wall -Wextra -Wno-unused-parameter`, own code) | 18 unique (F-020) |
| `ruff check python/` (pyproject rules) | 289 |
| pyflakes on tests, benchmarks, tools, scripts, demos | 87 |
| Broad `except` in package / silent `pass` | 257 / 126 |
| Doc pages with python blocks: run clean / fail | 107 / 69 (≈20 real, rest signature blocks) |
| `.pyi` names missing vs `__all__` | 71 of 137 |
| Parity vs NumPy (`benchmarks/r2_parity.py`, N=1024) | matmul 1.01×, solve 1.01×, invert 3.13×, det 3.43×, eigh 1.16×, **dot 0.47×, add 0.22×, multiply 0.29×** |

## How to reproduce

### Build (pip, the way users install)
```bash
apt-get install -y libopenblas-dev liblapacke-dev        # system BLAS/LAPACK
pip install "scikit-build-core>=0.3.3" "pybind11==2.12.0" setuptools_scm plotly psutil pytest pytest-timeout
pip install --no-build-isolation -v -Cbuild-dir=build/wheel .
```
Gotchas:
- **Sandboxed/cloud environments:** downloads from `github.com/.../archive/...` may be
  blocked by a proxy. Eigen comes from gitlab.com (works). googletest and the
  OpenBLAS fallback come from GitHub. For gtest use the distro source:
  `apt-get install googletest` and `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/usr/src/googletest`.
- The build prints `#pragma message: Spacetime.hpp included` (F-019) and a
  `_GNU_SOURCE redefined` warning. Both are harmless and pending removal.

### Python tests
The tests import from `python/` (sys.path hacks, F-023), so the compiled artifacts
must be copied next to the sources, exactly as CI does:
```bash
cp "$(python -c 'import pycauset,os;print(os.path.dirname(pycauset.__file__))')"/*.so python/pycauset/
python -m pytest tests/python -q -p no:cacheprovider --timeout=300 -rfEs
```
(Run the `cp` from a directory other than the repo root, or the source tree shadows
the installed package. The `*.so` files are gitignored.)

### C++ tests (with stricter warnings)
```bash
cmake -S . -B build/cpp -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -Dpybind11_DIR="$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')" \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/usr/src/googletest \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wno-unused-parameter"
cmake --build build/cpp -j4 && (cd build/cpp && ctest --output-on-failure)
```
Orphaned tests, built by hand (pattern):
```bash
g++ -std=c++20 -O1 -fopenmp -Iinclude -I/usr/src/googletest/googletest/include \
  tests/test_op_registry.cpp -o build/cpp/test_op_registry -Lbuild/cpp -lpycauset_core \
  build/cpp/lib/libgtest_main.a build/cpp/lib/libgtest.a -lpthread -Wl,-rpath,$PWD/build/cpp
```
`test_parallel_utils` **hangs** in `NestedParallelism`. Run it with
`--gtest_filter=-*Nested*` or under `timeout`.

### ASan + UBSan
```bash
cmake -S . -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -Dpybind11_DIR=... \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build/asan -j4
# stage an instrumented copy of the package
S=/tmp/asanpkg; mkdir -p $S && cp -r python/pycauset $S/ && rm -f $S/pycauset/*.so \
  && cp build/asan/_pycauset*.so build/asan/libpycauset_core.so $S/pycauset/
LD_PRELOAD="$(find /usr/lib -name 'libstdc++.so.6'|head -1):$(gcc -print-file-name=libasan.so):$(gcc -print-file-name=libubsan.so)" \
ASAN_OPTIONS=detect_leaks=0:verify_asan_link_order=0 UBSAN_OPTIONS=print_stacktrace=1 OPENBLAS_NUM_THREADS=1 \
python -c "import sys; sys.path.insert(0,'$S'); import pycauset, pytest; sys.exit(pytest.main(['tests/python','-v','--timeout=900']))"
```
Importing pycauset from the staged copy *before* pytest runs makes the tests' own
`sys.path` insertions harmless, because the module is already cached. Add
`-fno-sanitize-recover=undefined` to make UB fatal (the CI recommendation).

### Probes (review/probes/)
```bash
PYTHONPATH=python python review/probes/dtype_matrix_probe.py --verbose   # dtype × op vs NumPy
PYTHONPATH=python python review/probes/numpy_fallback_probe.py           # which ops call numpy.linalg
PYTHONPATH=python python review/probes/out_of_core_probe.py              # forced-disk correctness
PYTHONPATH=$PWD/python python review/probes/run_doc_snippets.py          # execute doc examples
```
⚠ **Disk:** doc examples allocate large matrices (one asks for a 1M-element causal
set, about 125 GB). The snippet runner caps files at 1 GiB per process
(RLIMIT_FSIZE) and deletes each page's working directory. Don't run doc examples
without it. In session 1 an uncapped run filled the disk, which also killed a
parallel build.

### Coverage (Python + C++) and dead-code tools
See DEADCODE.md §2 for the full pipeline. The C++ coverage build:
```bash
cmake -S . -B build/cov -G Ninja -DCMAKE_BUILD_TYPE=Debug -Dpybind11_DIR=... \
  -DCMAKE_CXX_FLAGS="--coverage -O0" -DCMAKE_SHARED_LINKER_FLAGS=--coverage -DCMAKE_MODULE_LINKER_FLAGS=--coverage
cmake --build build/cov -j3      # ~15 min; the largest TU needs ~9 GB RAM
```
Stage it like the ASan build (copy `.so` files into a package copy, import it before
pytest). Under coverage the suite takes ~4 min instead of 21 s.

### Lint / warnings
```bash
ruff check python/ --statistics
ruff check tests benchmarks tools scripts demos --select F --statistics
pip install vulture && vulture python/pycauset --min-confidence 60   # dead-code hints (noisy on public API)
```

### Performance
```bash
PYTHONPATH=python python benchmarks/r2_parity.py
```
Note: on this 4-vCPU container NumPy's own `solve(1024)` takes ~0.7 s (it's slow
regardless of pycauset), so absolute times are only comparable on the same machine.
Use the ratios.
