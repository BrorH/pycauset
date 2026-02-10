# pinning_budget

Pinned-memory budget (bytes) used by the `MemoryGovernor` for GPU transfers.

By default, PyCauset uses a dynamic heuristic:

$$
\text{Budget} = \min(0.5\cdot\text{SystemRAM},\ 0.8\cdot\text{FreeRAM},\ 8\text{GB})
$$

You can override this with `pycauset.cuda.set_pinning_budget(bytes)`.

## See also

- [[docs/functions/pycauset.cuda.set_pinning_budget.md|pycauset.cuda.set_pinning_budget]]
- [[internals/MemoryArchitecture.md|MemoryArchitecture]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[guides/Performance Guide.md|Performance Guide]]
