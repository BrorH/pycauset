# pycauset.cuda.benchmark

```python
pycauset.cuda.benchmark(force: bool = False) -> dict | None
```

Runs the GPU hardware audit and micro-benchmarks used by the dispatcher.

The benchmark probes PCIe bandwidth and GPU GEMM throughput (SGEMM/DGEMM) and caches results to `~/.pycauset/hardware_profile.json`. If no GPU is active, returns `None`.

## Parameters

*   **force** (*bool*): If `True`, re-run the benchmarks even if a cached profile exists.

## Returns

*   **dict | None**: A dictionary with benchmark results (or `None` if no GPU is active).

## Behavior

*   If a valid cached profile exists and `force=False`, returns quickly without re-running benchmarks.
*   If the cached profile is missing or incompatible, the benchmarks are executed.
*   Updates the in-memory hardware profile used by the cost model.

## Exceptions / warnings

*   Benchmarks can take a few seconds on first run.
*   Returns `None` if CUDA is not active.

### Returned keys

*   **device_id** (*int*)
*   **device_name** (*str*)
*   **cc_major** (*int*)
*   **cc_minor** (*int*)
*   **pci_bandwidth_gbps** (*float*)
*   **sgemm_gflops** (*float*)
*   **dgemm_gflops** (*float*)
*   **timestamp_unix** (*int*)

## Notes

*   `pycauset.cuda.enable()` must succeed before this returns a profile.
*   To reset the cache, delete `~/.pycauset/hardware_profile.json`.

## Example

```python
import pycauset as pc

info = pc.cuda.benchmark(force=True)
print(info)

if info is None:
    pc.cuda.enable()
    info = pc.cuda.benchmark(force=True)
    print(info)
```

## See also

- [[docs/functions/pycauset.cuda.force_backend.md|pycauset.cuda.force_backend]]
- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.is_available.md|pycauset.cuda.is_available]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[guides/Performance Guide.md|Performance Guide]]
