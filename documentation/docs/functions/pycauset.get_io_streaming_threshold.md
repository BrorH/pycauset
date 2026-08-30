# pycauset.get_io_streaming_threshold

```python
pycauset.get_io_streaming_threshold() -> int
```

Return the current IO routing threshold in bytes.

This threshold drives the IO-observability routing heuristics: operations whose
estimated working set is below it are routed "direct", those above it "streaming".
It is a heuristic for observability and routing, separate from
[[pycauset.get_memory_threshold|pycauset.get_memory_threshold]] (which controls
RAM vs disk backing).

## Returns

*   **int**: The current IO routing threshold in bytes.
