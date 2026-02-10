# Performance and Acceleration Guide

PyCauset is designed to automatically optimize performance based on your hardware using a unified **Compute Context**. This guide explains how the GPU acceleration works, how precision (Float32 vs Float64) is handled, and how to achieve maximum throughput.

## 1. GPU Acceleration

PyCauset includes a high-performance GPU backend powered by NVIDIA CUDA. This backend is managed by the **ComputeContext** and is designed to be **frictionless**: it requires no configuration, automatically detects compatible hardware, and seamlessly falls back to the CPU if no GPU is available.

### Features

*   **Automatic Detection**: The ComputeContext checks for a CUDA-capable GPU at startup.
*   **Dynamic Loading**: The GPU backend is a separate plugin (`pycauset_cuda.dll` / `.so`). You do not need CUDA installed to run PyCauset on a CPU-only machine.
*   **Out-of-Core Processing**: Algorithms are designed to handle matrices larger than your GPU's VRAM by streaming data efficiently between Disk, RAM, and VRAM.

### Requirements

To use the GPU acceleration, you need:

1.  **Hardware**: An NVIDIA GPU with Compute Capability 6.0 or higher (Pascal or newer).
2.  **Drivers**: Up-to-date NVIDIA Drivers.
3.  **Software**: The `pycauset_cuda` plugin must be present in the library directory (installed automatically if built with CUDA support).

### Supported Operations

The following operations are currently accelerated via the AutoSolver:
*   **Matrix Multiplication** (`matmul`, `*` operator):
    *   Uses `cuBLAS` for high-performance GEMM (Float32/Float64).
    *   **BitMatrices**: `DenseBitMatrix` and `TriangularBitMatrix` multiplication is **fully accelerated** using custom bit-packed kernels. This allows for extremely fast path counting and transitive closure operations (up to 64x faster than float operations).
*   **Matrix Inversion** (`inverse`):
    *   Uses `cuSOLVER` LU factorization.

### Streaming constraints

GPU acceleration assumes **streaming-first** execution. Operations that exceed VRAM are tiled and pipelined so the CPU can prepare the next batch while the GPU computes.

**Constraints to know:**
- Routing may prefer CPU for medium sizes if PCIe transfer cost dominates.
- Pinned memory is budgeted; if the pinning budget is exhausted, transfers degrade to pageable memory (slower but safe).
- Some operations remain CPU-only until GPU kernels are implemented (routing is explicit and testable).

#### Minimal example (force streaming decisions)

```python
import pycauset as pc

# Lower the IO streaming threshold to force streaming routing.
pc.set_io_streaming_threshold(64)  # bytes

A = pc.Matrix(128, 128)
B = pc.Matrix(128, 128)
C = A @ B
```

#### Practical example (disk-backed matrices)

```python
import pycauset as pc
from pathlib import Path

pc.set_backing_dir(Path("./storage"))
pc.set_memory_threshold(256 * 1024 * 1024)  # 256 MB

# This forces a file-backed matrix on most machines.
A = pc.FloatMatrix(50000)
A.set_identity()

B = pc.FloatMatrix(50000)
B.fill(1.0)

C = A @ B  # streaming route if thresholds are exceeded
```

### Routing policy and control

PyCauset routes operations through a cost model that weighs PCIe transfer time against GPU compute throughput. The first time a GPU is used, PyCauset runs a small benchmark and caches it to `~/.pycauset/hardware_profile.json`.

#### Minimal example

```python
import pycauset as pc

pc.cuda.enable()
pc.cuda.benchmark(force=False)  # warm the hardware profile cache
```

#### Practical example (deterministic routing)

```python
import pycauset as pc

pc.cuda.enable()
pc.cuda.force_backend("cpu")  # force CPU for validation
C_cpu = pc.Matrix(2048, 2048) @ pc.Matrix(2048, 2048)

pc.cuda.force_backend("gpu")  # prefer GPU for throughput
C_gpu = pc.Matrix(2048, 2048) @ pc.Matrix(2048, 2048)

pc.cuda.force_backend("auto")
```

## 2. Precision and Types

PyCauset respects the user's choice of type but employs smart defaults to maximize performance on consumer hardware.

### Automatic Hardware Detection

When PyCauset initializes, the ComputeContext queries your system:

*   **CPU**: Detects AVX2/AVX-512 support for vectorized operations.
*   **GPU**: Checks for CUDA-capable devices.
    *   **Consumer GPUs (GeForce GTX/RTX)**: Typically default to **Float32** due to hardware constraints (poor Float64 performance).
    *   **Data Center GPUs (Tesla/A100)**: May default to **Float64** if supported.

### Default Behavior

When you convert a NumPy array to a PyCauset matrix using `pycauset.matrix()`:

1.  **Float32 Input**: Always creates a `Float32Matrix`.
2.  **Float64 Input (Standard NumPy)**:
    *   If your hardware prefers Float32 (e.g., GTX 1060), PyCauset will **automatically downcast** to `Float32Matrix` for a 15-30x speedup.
    *   If your hardware supports fast Float64, it creates a `Float64Matrix`.

### Benchmarks (Example: GTX 1060)

*   **Float64**: ~135 GFLOPS
*   **Float32**: ~4400 GFLOPS (**32x faster**)

PyCauset ensures you get this speedup by default.

# 3. Where PyCauset is Faster than NumPy

PyCauset isn’t faster at everything, but it is **decidably faster** in specific large-scale regimes.

### 1. Out-of-Core Processing (Infinite Speedup)
Standard NumPy crashes with `MemoryError` if a matrix exceeds RAM.
*   **PyCauset**: Transparently tiles operations on disk-backed matrices (up to Terabytes).
*   **Result**: Infinite speedup (vs crash).

### 2. High-Bandwidth I/O (up to 10x faster)
PyCauset uses a multithreaded "Direct Path" for loading/saving data and importing/exporting from NumPy, bypassing python loop overhead.
*   **Contiguous Write**: **~10.0x faster** than standard `np.tofile` or pickle for large buffers (>1GB).
*   **Import**: **~2.7x faster** for importing non-contiguous (sliced) NumPy arrays.

### 3. Bit-Packed Boolean Math (up to 64x faster)
NumPy stores booleans as 8-bit bytes (`bool_`). PyCauset stores them as 1-bit packs (`DenseBitMatrix`).
*   **Storage**: 8x smaller RAM footprint.
*   **Matmul**: 64x faster (using specialized bit-block kernels).

### 4. Mixed-Precision GPU Offloading
NumPy is CPU-only. PyCauset seamlessly offloads large `matmul` or `inverse` ops to CUDA without the user managing device memory.
*   **Latency**: Lower for small ops (don't use GPU for 10x10 matrices).
*   **Throughput**: Orders of magnitude higher for large matrices (e.g., 4000x4000).

## 4. Memory Management Strategy

PyCauset uses a **"RAM-First"** architecture to maximize speed.

1.  **Use All Available RAM**: The system aggressively utilizes available physical RAM to keep matrices in memory for maximum throughput.
2.  **Automatic Disk Spillover**: When physical RAM is exhausted, the system can spill by switching to file-backed (memory-mapped) storage (for example `.tmp` backing files under the backing directory).
3.  **Hybrid Async Pipeline**:
    *   **Small Matrices**: If a matrix fits entirely in VRAM, it is uploaded once and processed at maximum speed.
    *   **Large Matrices**: If a matrix exceeds VRAM (or the configured memory_limit), the system automatically switches to **Streaming Mode**.
        *   **CPU Parallelism**: The CPU uses multi-threading to pack data into pinned memory buffers.
        *   **GPU Overlap**: The GPU computes the current chunk while the CPU prepares the next one.

## 5. GPU Configuration

While PyCauset works out-of-the-box, power users can fine-tune the behavior through the ComputeContext (exposed via `pycauset.cuda` for backward compatibility):

```python
import pycauset.cuda

# Enable with specific settings
pycauset.cuda.enable(
    device_id=0,              # Select specific GPU
    memory_limit=4*1024**3,   # Limit VRAM usage to 4GB
    enable_async=True         # Enable/Disable Async Pipelining
)

# Optional: override the pinned-memory budget (bytes)
pycauset.cuda.set_pinning_budget(2 * 1024**3)
```

Additional controls:

```python
import pycauset as pc

pc.cuda.is_available()        # True/False
pc.cuda.current_device()      # Human-readable device label
pc.cuda.force_backend("gpu")  # Force GPU routing (or "cpu", "auto")
pc.cuda.benchmark(force=True) # Recompute hardware profile + cache
pc.cuda.disable()             # Disable GPU routing
```

Hardware profile cache:

- Stored at `~/.pycauset/hardware_profile.json`.
- Delete the file to force a clean re-benchmark.
- `pc.cuda.benchmark(force=True)` always regenerates the profile.

## 6. Troubleshooting

If you believe you have a GPU but PyCauset is using the CPU:

1.  Check if `pycauset.cuda.is_available()` returns `True`.
2.  Ensure `pycauset_cuda.dll` (Windows) or `libpycauset_cuda.so` (Linux) exists in the installation folder.
3.  Verify your NVIDIA drivers are installed and working (run `nvidia-smi`).

## See also

- [[docs/functions/pycauset.cuda.enable.md|pycauset.cuda.enable]]
- [[docs/functions/pycauset.cuda.force_backend.md|pycauset.cuda.force_backend]]
- [[docs/functions/pycauset.cuda.benchmark.md|pycauset.cuda.benchmark]]
- [[docs/functions/pycauset.cuda.set_pinning_budget.md|pycauset.cuda.set_pinning_budget]]
- [[docs/functions/pycauset.set_backing_dir.md|pycauset.set_backing_dir]]
- [[docs/functions/pycauset.set_memory_threshold.md|pycauset.set_memory_threshold]]
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[internals/Streaming Manager.md|Streaming Manager]]

## 7. Lazy Evaluation

PyCauset employs **Lazy Evaluation** to optimize complex mathematical expressions. When you perform operations like `C = A + B`, the result is not computed immediately. Instead, a lightweight "Expression" object is created.

### How it Works

*   **Fusion**: Operations are fused together. `D = A + B + C` is computed in a single pass, avoiding the creation of a temporary matrix for `A + B`. This reduces memory bandwidth usage by 50% or more.
*   **Just-In-Time**: Computation happens only when you access the data (e.g., `print(C[0,0])`) or explicitly request it (e.g., `C.eval()`).
*   **Memory Efficiency**: Since intermediate results are not materialized, you can perform complex chains of operations on matrices that would otherwise exceed your RAM if all intermediates were stored.

### Manual Control

While the system handles this automatically, you can force evaluation or manage memory manually:

*   **`.eval()`**: Forces a lazy expression to compute and return a materialized matrix.
    ```python
    expr = A + B
    result = expr.eval() # 'result' is a real Matrix
    ```
*   **`spill_to_disk()`**: If you have a large materialized matrix in RAM and need to free up space for other computations without deleting the data, you can force it to move to disk.
    ```python
    A = pycauset.FloatMatrix(10000, 10000)
    # ... fill A ...
    A.spill_to_disk() # Moves A's data to a temp file, freeing RAM
    ```

