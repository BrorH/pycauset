# pycauset.MemoryHint

```python
pycauset.MemoryHint
```

A memory-access hint for a file-backed object: the region and order in which its storage
will be read, passed with `matrix.hint(h)`. The IO accelerator reads `Sequential` and
`Strided` regions ahead (`madvise(MADV_WILLNEED)` / `PrefetchVirtualMemory`); the other
patterns are accepted and do nothing yet. No engine operation issues hints itself today.

Build one with the static constructors `MemoryHint.sequential(start, len)` and
`MemoryHint.strided(start, len, stride, block)`.

## Attributes

*   **start_offset** (*int*): Byte offset of the start of the hinted region.
*   **length** (*int*): Byte length of the hinted region.
*   **pattern** (*pycauset.AccessPattern*): The declared access order.
*   **block_bytes** (*int*): Block size for blocked access.
*   **stride_bytes** (*int*): Stride between accesses when the pattern is strided.

## See also

*   [[docs/classes/core/pycauset.AccessPattern.md|pycauset.AccessPattern]]
*   [[internals/Out of Core.md|Out of Core]]
