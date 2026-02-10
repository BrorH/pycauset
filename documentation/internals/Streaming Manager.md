# Streaming Manager

The streaming manager is the shared policy engine for out-of-core execution. It decides when to stream, how to tile, how deep to queue, and records what happened so tests and users can see the plan. Matmul, invert, eigvalsh, eigh, and eigvals_arnoldi register descriptors that plug in access patterns, guards, and resource budgets.

This document is for contributors who need to extend streaming behavior or debug routing decisions.

## Algorithm Drivers (Host-Orchestrated)

For GPU acceleration, PyCauset uses **Algorithm Drivers**: host-side control loops that orchestrate the low-level streaming pipeline without embedding device-specific logic. Drivers are responsible for sequencing tiles, enforcing dependencies, and coordinating the CPU/GPU overlap. The device-specific worker (CUDA today, CPU later) only executes kernels on already-prepared tiles.

### Why drivers exist
- **Streaming-first**: Most target workloads do not fit in VRAM (or RAM), so drivers must stream by design.
- **Plug-and-play**: Adding a new accelerated op should not require deep CUDA or AVX knowledge.
- **CPU/GPU parity**: The same driver contract is used by both the CUDA worker and the R1_CPU worker pipeline.

### Driver contract (summary)
- **Inputs**: operand metadata, tile shapes, and a streaming plan (queue depth, access pattern).
- **Outputs**: explicit success/failure + trace annotations (no silent fallbacks).
- **Budget discipline**: pinned allocations require a ticket from `MemoryGovernor`.
- **Observability**: emit deterministic trace tags for routing and pipeline stages.

### Where drivers live
- C++ driver classes live alongside device workers (e.g., `src/accelerators/cuda/` or `src/compute/cpu/`), but remain device-agnostic where possible.
- AutoSolver owns routing decisions and calls into the driver (via `ComputeWorker`) once the device is selected.

### Driver failure modes (expected)
- **Unsupported dtype/structure**: driver must return a clear failure reason so AutoSolver can fall back.
- **Budget denial**: if pinned memory cannot be acquired, drivers must degrade to pageable buffers or abort with a routed fallback.
- **Non-streamable shapes**: guards must route to `direct` with a deterministic reason.

## What the manager owns
- Routing: picks `route` (`streaming` or `direct`) and `reason`, combining IO observability thresholds with per-op guards.
- Planning: fills plan fields (`tile_shape`, `queue_depth`, `plan.access_pattern`, `trace_tag`, `events`, `storage` summary).
- Execution glue: best-effort prefetch/discard for streaming routes and `impl=...` annotations for the chosen implementation.
- Registry: per-op `StreamingDescriptor` entries that provide access patterns and policy hooks.

## Lifecycle at a glance
1. **plan(op, operands, allow_huge=False)**: snapshots operands, chooses route, runs the descriptor guard, and computes tiles/queue depth.
2. **prefetch(plan, operands)**: only when `route == "streaming"`; defaults to calling accelerators on backing files.
3. **compute**: the op executes (native, Python fallback, or BlockMatrix orchestration). The manager can annotate the implementation (`impl=...`).
4. **discard(plan, operands, result)**: only when streaming; best-effort discard on backing ranges.
5. **inspect**: `pc.last_io_trace(...)` returns the plan and event timeline for debugging and tests.

## Plan schema (recorded via IO observability)
- `route` / `reason`: routing choice and justification.
- `tile_shape`: `(rows, cols)` when streaming; `None` when direct. Tiles clamp to operand shapes.
- `queue_depth`: bounded to `[1, 8]` when streaming; `0` when direct.
- `plan.access_pattern`: descriptor-supplied tag (e.g., `blocked_rowcol`).
- `trace_tag`: monotonic tag per op (`op:N`).
- `events`: list of `{type, detail, reason?}`; includes plan/prefetch/discard/compute annotations.
- `storage`: backing files, temporary flags, and storage roots gathered from operand snapshots.

## Routing rules
- **File-backed operands**: force `streaming` with reason `file-backed operand`.
- **allow_huge=True**: force `direct` with reason `allow_huge bypassed threshold`.
- **Threshold set**: any estimated operand bytes over threshold → `streaming` with `estimated bytes exceed threshold`.
- **Threshold None**: `direct` with `no threshold configured` unless a guard overrides.
- **Guards**: per-op hooks can override route/reason. Example: non-square invert/eig* → `direct` with `non_square`; matmul mismatched shapes → `direct` with `shape_mismatch`.

## Descriptor catalog (current)
| op | access_pattern | guard | tile budget | queue depth | notes |
| --- | --- | --- | --- | --- | --- |
| matmul | blocked_rowcol | shape mismatch → direct | budgeted square tiles clamped to shapes | 3 when streaming | Python tiling fallback annotates `impl=streaming_python` |
| invert | invert_dense | non-square → direct | default square tile | 1 when streaming | Fallback annotates `impl=streaming_python` |
| eigvalsh | symmetric_eigvals | non-square → direct | default square tile | 1 when streaming | Fallback annotates `impl=streaming_python` |
| eigh | symmetric_eigh | non-square → direct | default square tile | 1 when streaming | Fallback annotates `impl=streaming_python` |
| eigvals_arnoldi | arnoldi_topk | non-square → direct | default square tile | 1 when streaming | Fallback annotates `impl=streaming_python` |

## Hooks and defaults
- `tile_budget_fn(threshold_bytes, snapshots)`: derives tiles from the memory threshold and itemsize; matmul halves budget across A/B and clamps to shapes; square ops reuse the default derivation.
- `queue_depth_fn(route, snapshots)`: returns depth before coercion; manager caps to `[1, 8]` and zeroes when direct.
- `guard(operands, snapshots, allow_huge)`: may override route/reason early; does not materialize data, uses snapshots.
- `prefetch` / `discard`: optional per-op hooks; defaults call accelerator prefetch/discard on backing files.
- `annotate_impl(record, label)`: attaches `impl=label` and records a compute event.

## Safety guarantees
- No streaming queue depth above 8; non-streaming queues are 0.
- Tile shapes always finite and clamped to operand extents; failures fall back to conservative defaults.
- Guards run before tiling so invalid shapes revert to direct routes instead of crashing in streaming codepaths.
- IO observability storage summaries remain intact for spill/backing-file diagnostics.

## Debugging and tests
- `pc.last_io_trace()` shows the latest plan; `pc.last_io_trace("matmul")` fetches by op.
- Event timeline should contain `plan` plus `io` (prefetch/discard) and `compute` (impl) entries for streaming routes.
- Threshold-driven scenarios: set `pc.set_io_streaming_threshold(bytes)` to force streaming in tests; set to `None` to validate the direct path.
- File-backed fakes should force streaming regardless of threshold, exercising the guardrail for spill-backed inputs.

## Where this lives in code

- Policy + routing: [python/pycauset/_internal/streaming_manager.py](python/pycauset/_internal/streaming_manager.py)
- IO trace helpers: [python/pycauset/_internal/io_observability.py](python/pycauset/_internal/io_observability.py)
- Streaming integrations in ops: [python/pycauset/_internal/ops.py](python/pycauset/_internal/ops.py)
- C++ Implementation (for native CPU backend):
    - [include/pycauset/compute/StreamingManager.hpp](include/pycauset/compute/StreamingManager.hpp)
    - [src/compute/StreamingManager.cpp](src/compute/StreamingManager.cpp)

## Extending (safe checklist)

1. Add a `StreamingDescriptor` with a conservative guard.
2. Provide a tile budget function that clamps to operand shapes.
3. Add a queue-depth function and keep it in `[1, 8]`.
4. Ensure all annotations are deterministic (no timing-dependent fields).
5. Add a routing test that asserts the `route` and `reason` fields.

### Extension guidelines
- Add a new op by registering a `StreamingDescriptor` in `pycauset.__init__` with an access pattern, guard, and budget/queue hooks.
- Keep guards deterministic and non-materializing (only consult snapshots and metadata).
- Prefer small, conservative tile/queue defaults; tighten once end-to-end tests validate throughput.

## See also

- [[project/protocols/Adding Operations.md|Adding Operations protocol]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[guides/Performance Guide.md|Performance Guide]]
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
- [[docs/functions/pycauset.set_memory_threshold.md|pycauset.set_memory_threshold]]
