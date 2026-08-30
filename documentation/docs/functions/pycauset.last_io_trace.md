# pycauset.last_io_trace

```python
pycauset.last_io_trace(op: str | None = None) -> dict | None
```

Return the most recent IO trace, optionally filtered by operation name.

Each trace records the chosen route (direct vs streaming), the reason, operand
storage summaries, tile shape, queue depth, and events. Useful for answering "did
this op spill to disk, and why".

## Parameters

*   **op** (*str, optional*): Filter to the most recent trace for this operation
    name (for example `"matmul"`). When `None`, returns the latest trace regardless
    of operation.

## Returns

*   **dict | None**: The most recent trace record, or `None` when no matching trace
    exists.

## See also

*   [[pycauset.clear_io_traces|pycauset.clear_io_traces]]
*   [[pycauset.set_io_streaming_threshold|pycauset.set_io_streaming_threshold]]
