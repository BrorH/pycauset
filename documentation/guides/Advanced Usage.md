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

pc.set_memory_threshold(100 * 1024 * 1024)   # spill above 100 MB
print(pc.get_memory_threshold())             # read it back
pc.set_memory_threshold(None)                # reset to the default
```

## 2. IO streaming threshold

Separately, the IO observability layer routes operations through streaming (out-of-core)
heuristics above a byte threshold. This is a routing hint, not the RAM/disk cutoff.

```python
pc.set_io_streaming_threshold(1024 * 1024 * 1024)   # 1 GB
print(pc.get_io_streaming_threshold())
pc.set_io_streaming_threshold(None)                 # automatic
```

## 3. Backing directory

Disk-backed payloads live in a `.pycauset` directory under the working directory by
default. Point it somewhere else once, right after import, before you allocate
anything large.

```python
from pathlib import Path
import pycauset as pc

pc.set_backing_dir(Path.cwd() / "pycauset_storage")
```

Changing it after matrices exist is allowed but not guaranteed to be clean; you get a
`PyCausetStorageWarning` if live matrices are still tracked.

## 4. Thread count

```python
pc.set_num_threads(4)        # cap parallel work at 4 threads
print(pc.get_num_threads())
```

The default is the machine's CPU count. `pc.configure_openblas_threads()` reapplies
the current thread setting to the OpenBLAS pool if the build uses it.

## 5. Backend override (debug)

The autosolver normally routes each operation to CPU or GPU by cost. You can force a
device. `pycauset.cuda` is safe to import on a CPU-only install: the controls are
no-ops when no CUDA device is present.

```python
pc.cuda.is_available()      # bool
pc.cuda.force_backend("cpu")  # or "gpu" (raises if unavailable), "auto"
```

## 6. IO trace observability

The IO layer records routing decisions. Read the most recent one, or filter by
operation name, when you want to see why something streamed or stayed in RAM.

```python
pc.last_io_trace()          # most recent trace, or None
pc.last_io_trace("matmul")  # most recent matmul trace
pc.clear_io_traces()        # reset the log
```

## 7. Keeping temporary files

Temporary backing files are deleted on exit by default. Set the flag to keep them for
debugging.

```python
pc.keep_temp_files = True
```

## 8. Precision mode

Temporarily override the promotion precision mode with a context manager:

```python
with pc.precision_mode("highest"):
    C = A @ B
```

This controls storage-dtype promotion decisions, not the accelerator's internal
compute dtype.

See [[docs/index|API Reference]] for exact signatures.
