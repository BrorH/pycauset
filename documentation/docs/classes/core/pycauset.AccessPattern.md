# pycauset.AccessPattern

```python
pycauset.AccessPattern
```

Enum describing how a streaming operation walks a backing buffer.

Used by memory hints and the streaming manager to declare the intended access
order so lookahead and prefetch can be scheduled accordingly.

## Members

*   **Sequential**: forward, one element after another.
*   **Reverse**: backward through the buffer.
*   **Random**: no predictable order.
*   **Strided**: fixed stride between accesses.
*   **Once**: a single pass with no reuse.

## See also

*   [[docs/classes/core/pycauset.MemoryHint.md|pycauset.MemoryHint]]
*   [[internals/Streaming Manager.md|Streaming Manager]]
