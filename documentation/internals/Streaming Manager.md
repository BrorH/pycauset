# Streaming Manager

The StreamingManager (`src/compute/StreamingManager.cpp`) runs an operation tile by tile when
its operands and result do not fit in RAM. It is part of the engine; there is no Python
routing layer (review F-100 removed one that guessed a route the engine never read).

This page is for contributors who extend out-of-core execution or debug it.

## When the engine streams

`CpuDevice` (`src/compute/cpu/CpuDevice.cpp`) decides per call:

1. It estimates the bytes the call touches (operands + result).
2. It asks the MemoryGovernor whether that fits the RAM budget (free RAM minus a safety
   margin: `can_fit_in_ram`).
3. If it does not fit **and** the operands qualify, it runs the StreamingManager; otherwise the
   direct kernel, which still works on file-backed (memory-mapped) storage through paging.

Operands qualify when all three matrices are plain dense views of one dtype: no
transposition, window, scale factor or conjugation (the tiles are copied from raw row-major
storage, F-040).

| Operation | Streamed dtypes | Kernel label |
|---|---|---|
| `matmul` | float32, float64 | `streaming.matmul` |
| `+ - *` and `/` of same-dtype operands | float32, float64, int32, int64 | `streaming.elementwise` |

Every other operation runs its direct kernel. LAPACK-based decompositions (`eigh`, `eig`,
`svd`, `qr`, `cholesky`, `lu`, `solve`, `inv`, `det`) need the whole matrix and do not stream.
`eigvals_arnoldi` reads the matrix only through matrix-vector products, so it works on
file-backed matrices without materialising them.

## Tiles

`StreamingManager::calculate_block_size` sizes square tiles so that three of them (A, B and C)
use at most 80% of the RAM the MemoryGovernor reports as available. A tile is at least
4096 × 4096 elements (or the whole matrix, if smaller), rounded down to a multiple of 8.

## Seeing what happened

- `pc.last_io_trace(op)` lists the kernels the engine ran for the last traced call and sets
  `streamed` when one of them was a `streaming.*` kernel
  ([[docs/functions/pycauset.last_io_trace.md|pycauset.last_io_trace]]).
- `pc._native._debug_last_kernel_trace()` is the label of the last kernel that ran.
- Tests force the streaming path with a huge RAM safety margin, so that nothing "fits":
  `pc._native._set_ram_safety_margin(1 << 62)` (see `tests/engine/test_streaming.py`).

## Where this lives in code

- Decision: `src/compute/cpu/CpuDevice.cpp` (`can_stream_*`, the MemoryGovernor checks)
- Tiling: `include/pycauset/compute/StreamingManager.hpp`, `src/compute/StreamingManager.cpp`
- Tile kernels: `src/compute/cpu/CpuComputeWorker.cpp` (same contract as the dense kernels:
  same dtype, plain dense)
- Budget: `src/core/MemoryGovernor.cpp`

## See also

- [[project/protocols/Adding Operations.md|Adding Operations protocol]] (§7 Streaming)
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[guides/Storage and Memory|Storage and Memory]]
- [[docs/functions/pycauset.set_memory_threshold.md|pycauset.set_memory_threshold]]
