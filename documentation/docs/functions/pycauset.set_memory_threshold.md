# pycauset.set_memory_threshold

```python
pycauset.set_memory_threshold(bytes: int)
```

Sets the size threshold (in bytes) below which objects are stored in RAM instead of on disk.

Objects smaller than this threshold will be created in memory (using anonymous mapping) to improve performance and avoid disk I/O. Objects larger than this threshold will be backed by temporary files on disk.

The default threshold is 1 GB.

## Parameters

*   **bytes** (*int*): The threshold in bytes.

## Example

```python
import pycauset

# Set threshold to 100 MB
pycauset.set_memory_threshold(100 * 1024 * 1024)

# This matrix (approx 12.5 MB) will now be in RAM
m = pycauset.TriangularBitMatrix(10000) 
```
