# Streaming Manager

The streaming manager is the shared policy engine for out-of-core execution. It decides when to stream, how to tile, how deep to queue, and records what happened so tests and users can see the plan. Matmul, invert, eigvalsh, eigh, and eigvals_arnoldi register descriptors that plug in access patterns, guards, and resource budgets.

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

## Extending
- Add a new op by registering a `StreamingDescriptor` in `pycauset.__init__` with an access pattern, guard, and budget/queue hooks.
- Keep guards deterministic and non-materializing (only consult snapshots and metadata).
- Prefer small, conservative tile/queue defaults; tighten once end-to-end tests validate throughput.
