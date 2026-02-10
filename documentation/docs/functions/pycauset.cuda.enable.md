# pycauset.cuda.enable

```python
pycauset.cuda.enable(
    memory_limit: int | None = None,
    enable_async: bool = True,
    device_id: int = 0,
    stream_buffer_size: int = 64 * 1024 * 1024,
)
```

Enables the CUDA backend and configures GPU execution settings.

This call attempts to load the CUDA plugin and create a GPU device. If it
succeeds, the dispatcher can route eligible operations to GPU. If the plugin
cannot be loaded, the call is a no-op and routing remains CPU-only.

The first successful enable may also warm the hardware profile cache used by
the cost model (see `pycauset.cuda.benchmark`).

## Parameters

*   **memory_limit** (*int | None*): Maximum GPU memory to use (bytes). `None` uses auto-detection.
*   **enable_async** (*bool*): Enables asynchronous streaming (overlap transfers with compute).
*   **device_id** (*int*): GPU device index.
*   **stream_buffer_size** (*int*): Streaming buffer size (bytes) for tiled operations.

## Returns

*   **None**

## Behavior

*   Loads the CUDA plugin (if available) and creates a GPU device.
*   Makes the GPU available to the AutoSolver routing logic.
*   Does **not** force GPU usage; use `pycauset.cuda.force_backend("gpu")` for deterministic routing.
*   Repeated calls are allowed and can be used to change device/config settings.

## Exceptions / warnings

*   If the CUDA plugin is unavailable, this call is a no-op and GPU remains inactive.

## Example

```python
import pycauset as pc

pc.cuda.enable(memory_limit=4 * 1024 * 1024 * 1024, enable_async=True)

# Optional: bias routing to GPU for validation / benchmarking
pc.cuda.force_backend("gpu")
```

## See also

- [[docs/functions/pycauset.cuda.disable.md|pycauset.cuda.disable]]
- [[docs/functions/pycauset.cuda.is_available.md|pycauset.cuda.is_available]]
- [[docs/functions/pycauset.cuda.benchmark.md|pycauset.cuda.benchmark]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[guides/Performance Guide.md|Performance Guide]]
