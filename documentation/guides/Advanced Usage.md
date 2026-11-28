# Advanced Usage & Tuning

These are the knobs you reach for when you need to control memory, threads, storage
location, or the device backend. Most runs do not need any of them.

> **Warning**: These controls are for power users. Changing them can make things
> slower or less stable if you do not know why you are changing them.

## 1. Memory threshold (RAM vs disk)

PyCauset keeps small objects in RAM and moves large ones to disk-backed storage. The
cutoff is the native memory threshold, 1 GB by default.

```python
import pycauset as pc

pc.set_memory_threshold(100 * 1024 * 1024)   # file-backed above 100 MB
print(pc.get_memory_threshold())             # read it back
pc.set_memory_threshold(None)                # reset to the default
```

## 2. Backing directory

Disk-backed payloads live in a `.pycauset` directory under the working directory by
default. Point it somewhere else once, right after import, before you allocate
anything large.

```python
from pathlib import Path
import pycauset as pc

pc.set_backing_dir(Path.cwd() / "pycauset_storage")
```

Changing it later is allowed: objects already made keep their files in the old directory
(you get a `PyCausetStorageWarning` if there are any), and every directory used is swept at
exit.

## 3. Thread count

```python
pc.set_num_threads(4)        # cap parallel work at 4 threads
print(pc.get_num_threads())
```

The default is the machine's CPU count. `pc.configure_openblas_threads()` reapplies
the current thread setting to the OpenBLAS pool if the build uses it.

## 4. IO traces

Each traced `pc.<op>(...)` call records its operands (shape, dtype, storage) and the kernels
the engine ran, with each operand's storage (memory or file).

```python
pc.last_io_trace()          # most recent trace, or None
pc.last_io_trace("matmul")  # most recent matmul trace
pc.clear_io_traces()        # reset the log
```

## 5. Temporary files

A file-backed object's `.pycauset-tmp` file is deleted when the last object using it goes (views
share it). To keep a matrix, save it with `pc.save`.

See [[docs/index|API Reference]] for exact signatures.
