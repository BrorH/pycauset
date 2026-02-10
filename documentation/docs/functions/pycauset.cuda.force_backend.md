# pycauset.cuda.force_backend

```python
pycauset.cuda.force_backend(mode: str)
```

Forces the routing preference used by the compute dispatcher.

Use this to override the automatic CPU/GPU selection when you need deterministic behavior.

## Parameters

*   **mode** (*str*): One of:
    *   **"auto"** — use the cost model (default behavior).
    *   **"cpu"** — force CPU routing where possible.
    *   **"gpu"** — prefer GPU routing when supported.

## Returns

*   **None**

## Behavior

*   Applies process-wide and persists until changed.
*   Does not bypass op support checks; unsupported ops still fall back.
*   If CUDA is inactive, `"gpu"` behaves like `"cpu"`.

## Notes

*   Forcing **"gpu"** does not bypass type/operation support checks. Unsupported ops still fall back to CPU.
*   If no GPU is active, routing remains CPU-only.

## Exceptions / warnings

*   **ValueError**: If `mode` is not one of "auto", "cpu", or "gpu".

## Example

```python
import pycauset as pc

pc.cuda.force_backend("gpu")
# Run a large matmul with GPU routing when supported
C = pc.Matrix(4096, 4096) @ pc.Matrix(4096, 4096)

pc.cuda.force_backend("auto")

# Force CPU for regression tests
pc.cuda.force_backend("cpu")
```

## See also

- [[docs/functions/pycauset.cuda.benchmark.md|pycauset.cuda.benchmark]]
- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.set_pinning_budget.md|pycauset.cuda.set_pinning_budget]]
