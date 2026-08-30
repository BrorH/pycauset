# pycauset.set_io_streaming_threshold

```python
pycauset.set_io_streaming_threshold(bytes: int)
```

Set the routing threshold for IO observability heuristics, in bytes.

Operations whose estimated working set exceeds this threshold are routed through
the streaming path and recorded as such in [[pycauset.last_io_trace]]. Lowering it
(e.g. to a small value in a test) forces the streaming route; raising it keeps more
operations on the direct path.

## Parameters

*   **bytes** (*int*): The new IO routing threshold in bytes.

## See also

*   [[pycauset.get_io_streaming_threshold|pycauset.get_io_streaming_threshold]]
*   [[pycauset.set_memory_threshold|pycauset.set_memory_threshold]]
