# pycauset.MemoryHint

```python
pycauset.MemoryHint
```

A memory-access hint attached to a streaming operation.

The solver emits a hint describing the region and order in which a backing buffer
will be read, so the IO accelerator can prefetch and discard ranges ahead of use
(CCA lookahead).

## Attributes

*   **start_offset** (*int*): Byte offset of the start of the hinted region.
*   **length** (*int*): Byte length of the hinted region.
*   **pattern** (*pycauset.AccessPattern*): The declared access order.
*   **block_bytes** (*int*): Block size for blocked access.
*   **stride_bytes** (*int*): Stride between accesses when the pattern is strided.
*   **sequential** / **strided** (*bool*): Convenience flags derived from the pattern.

## See also

*   [[docs/classes/core/pycauset.AccessPattern.md|pycauset.AccessPattern]]
*   [[internals/Streaming Manager.md|Streaming Manager]]
