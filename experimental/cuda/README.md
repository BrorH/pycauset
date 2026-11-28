# experimental/cuda: the CUDA accelerator (on hold)

GPU acceleration is on hold (maintainer decision, review/RESTRUCTURE.md §3.1 item 3). The code
lives here so it stays out of the maintained tree, CI, the docs and agents' context, without
being lost. **Nothing here is built, tested or documented by default**, and it may not compile
against the current engine.

| path | what |
|---|---|
| `src/` | `pycauset_cuda` plugin: `CudaDevice`, `CudaSolver`, matmul/Cholesky/Arnoldi drivers |
| `bindings/bind_cuda.cpp` | `_pycauset_cuda` discovery module (`is_available`, `current_device`) |
| `bindings/bind_cuda_controls.cpp` | the former `pycauset.cuda` submodule (enable/disable/force_backend/benchmark/pinning budget), removed from `bind_core.cpp` |
| `python/` | the `pycauset.cuda` facade and the import-time GPU install hint |
| `tests/` | GPU and hybrid CPU/GPU tests from `tests/python/` |
| `docs/` | API pages for `pycauset.cuda.*`, `is_gpu_available`, `pinning_budget` |
| `CMakeLists.txt` | built only with `-DENABLE_CUDA=ON` (adds this directory) |

## The core's dispatch layer (removed, session 9)
The core no longer has a GPU hook. `ComputeContext` holds the `CpuSolver` directly (review 3.4
C-13). Removed with that change, all recoverable from git history (the commit "Remove the
forwarding device layers" on the review branch):
- `ComputeDevice` (`include/pycauset/compute/ComputeDevice.hpp`): the device interface.
- `AutoSolver` (`src/compute/AutoSolver.cpp`): the router with a GPU cost model, a hardware
  profile (`HardwareProfile.hpp`, `~/.pycauset/hardware_profile.json`) and pinned memory. Every
  method wrapped the GPU call in `try { gpu } catch { print; cpu }`, a silent fallback the
  Adding Operations protocol forbids (review 4.8): route by contract and let GPU errors surface.
- `CpuDevice`: one-line forwards to `CpuSolver`.
- `ComputeContext::try_load_cuda`: `dlopen("libpycauset_cuda.so")` at start, from the library
  search path including the working directory (review 3.4 C-14). A returning plugin should be
  loaded by absolute path from the core library's own directory.

## Bringing it back
Re-add `bind_cuda_controls(m)` to `bind_core.cpp`, move `python/cuda.py` back to
`python/pycauset/`, port to the current engine (structure classes, views per
review/features/LAZY.md), and restore the tests and docs.

The `pip install pycauset[gpu]` extra (NVIDIA runtime wheels `nvidia-cuda-runtime-cu12`,
`nvidia-cublas-cu12`, `nvidia-cusolver-cu12`, all `>=12.6`) was removed from `pyproject.toml`
along with the move; restore it with the plugin.
- The CPU device gained `lstsq` (session 9, `?gelsd`) and lost `compute_k`'s `num_threads`:
  `CudaDevice` needs both before it compiles again.
