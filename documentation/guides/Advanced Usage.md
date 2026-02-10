# Advanced Usage & Tuning

This guide covers advanced configuration knobs for performance tuning and debugging. 

> **Warning**: These controls are intended for power users. Altering them may degrade performance or cause stability issues in low-memory environments.

## 1. Streaming Threshold

PyCauset automatically decides when to stream data (out-of-core) versus computing directly in RAM. You can override the bytecode threshold that triggers streaming.

*   **API**: `pycauset.set_io_streaming_threshold(bytes: int | None)`
*   **Default**: `None` (Use internal MemoryGovernor heuristics, typically ~80% safety margin).
*   **Usage**:
    ```python
    import pycauset as pc
    # Force streaming for any op involving > 1GB data
    pc.set_io_streaming_threshold(1024 * 1024 * 1024)
    
    # Reset to automatic
    pc.set_io_streaming_threshold(None)
    ```

## 2. CPU Tile Size Override

Control the tile dimensions used by the CPU backend during streaming operations.

*   **API**: `pycauset.advanced.set_cpu_tile_size(size: int | tuple[int, int] | None)`
*   **Default**: `None` (Streaming Manager selects optimal size based on L3 cache and RAM).
*   **Usage**:
    ```python
    # Force 2048x2048 tiles
    pc.advanced.set_cpu_tile_size(2048)
    
    # Reset
    pc.advanced.set_cpu_tile_size(None)
    ```

## 3. Thread Count

Manually set the concurrency level for parallel operations (ParallelFor and OpenBLAS).

*   **API**: `pycauset.set_num_threads(n: int)`
*   **Default**: `os.cpu_count()`.
*   **Usage**:
    ```python
    # Limit to 4 threads
    pc.set_num_threads(4)
    ```

## 4. Backend Override (Debug)

Force the `AutoSolver` to route operations to a specific device, disregarding cost models.

*   **API**: `pycauset.cuda.force_backend(backend: str)`
*   **Values**: 
    *   `"auto"`: Smart routing (Default).
    *   `"cpu"`: Force CPU execution.
    *   `"gpu"`: Force GPU execution (raises error if unavailable).
*   **Usage**:
    ```python
    pc.cuda.force_backend("cpu")
    ```

## 5. Trace Verbosity

Control the detail level of the internal IO and kernel event tracing.

*   **API**: `pycauset.debug.set_trace_level(level: int)`
*   **Levels**: 
    *   0: None
    *   1: IO Trace (Routing decisions, plans).
    *   2: Kernel Trace (Micro-ops, tile execution).
*   **Usage**:
    ```python
    pc.debug.set_trace_level(1)
    # Check trace
    print(pc._debug_last_io_trace())
    ```
