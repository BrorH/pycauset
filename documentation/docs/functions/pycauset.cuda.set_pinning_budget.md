# pycauset.cuda.set_pinning_budget

```python
pycauset.cuda.set_pinning_budget(bytes: int)
```

Sets a manual pinned-memory budget for GPU transfers.

Pinned memory improves PCIe throughput but locks physical RAM. PyCauset enforces a budget via the `MemoryGovernor` to avoid OS instability. This call overrides the dynamic heuristic.

## Parameters

*   **bytes** (*int*): Maximum pinned host memory in bytes.

## Returns

*   **None**

## Behavior

*   Overrides the dynamic pinning heuristic for the remainder of the process.
*   Applies globally to all GPU transfers that use pinned host staging.

## Exceptions / warnings

*   Overriding the budget applies for the remainder of the process.
*   Setting this too high can destabilize the OS on memory-constrained systems.

## Example

```python
import pycauset as pc

# Limit pinned memory to 2GB
pc.cuda.set_pinning_budget(2 * 1024 * 1024 * 1024)

# Reduce to a more conservative budget
pc.cuda.set_pinning_budget(512 * 1024 * 1024)
```

## See also

- [[docs/parameters/pycauset.pinning_budget.md|pinning_budget]]
- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.benchmark.md|pycauset.cuda.benchmark]]
- [[guides/Performance Guide.md|Performance Guide]]
