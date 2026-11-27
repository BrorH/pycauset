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

## Hooks that remain in the core engine (dormant)
- `IComputeDevice` (`include/pycauset/compute/ComputeDevice.hpp`): the device interface.
- `ComputeContext::try_load_cuda` (`src/compute/ComputeContext.cpp`): at startup, tries to
  `dlopen` `pycauset_cuda` and call `create_cuda_device`. Without the plugin this is a no-op.
- `AutoSolver` GPU routing (`src/compute/AutoSolver.cpp`, `gpu_device_`): never taken without
  the plugin. Dead-code review (DEADCODE.md Tier B) decides whether these stay dormant or
  move here too. **Before the GPU path is re-enabled:** every AutoSolver method wraps the GPU
  call in `try { gpu } catch { print; cpu }`, a silent fallback the Adding Operations protocol
  forbids (review 4.8). Route by contract (dtype, structure, size) and let GPU errors surface.

## Bringing it back
Re-add `bind_cuda_controls(m)` to `bind_core.cpp`, move `python/cuda.py` back to
`python/pycauset/`, port to the current engine (structure classes, views per
review/features/LAZY.md), and restore the tests and docs.

The `pip install pycauset[gpu]` extra (NVIDIA runtime wheels `nvidia-cuda-runtime-cu12`,
`nvidia-cublas-cu12`, `nvidia-cusolver-cu12`, all `>=12.6`) was removed from `pyproject.toml`
along with the move; restore it with the plugin.
