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

## 2. Precision and Types

PyCauset respects the user's choice of type but employs smart defaults to maximize performance on consumer hardware.

### Automatic Hardware Detection

When PyCauset initializes, the ComputeContext queries your system:

*   **CPU**: Detects AVX2/AVX-512 support for vectorized operations.
*   **GPU**: Checks for CUDA-capable devices.
    *   **Consumer GPUs (GeForce GTX/RTX)**: Typically default to **Float32** due to hardware constraints (poor Float64 performance).
    *   **Data Center GPUs (Tesla/A100)**: May default to **Float64** if supported.

### Default Behavior

When you convert a NumPy array to a PyCauset matrix using `pycauset.asarray()`:

1.  **Float32 Input**: Always creates a `Float32Matrix`.
2.  **Float64 Input (Standard NumPy)**:
    *   If your hardware prefers Float32 (e.g., GTX 1060), PyCauset will **automatically downcast** to `Float32Matrix` for a 15-30x speedup.
    *   If your hardware supports fast Float64, it creates a `Float64Matrix`.

### Benchmarks (Example: GTX 1060)

*   **Float64**: ~135 GFLOPS
*   **Float32**: ~4400 GFLOPS (**32x faster**)

PyCauset ensures you get this speedup by default.

## 3. Memory Management Strategy

PyCauset uses a **"RAM-First"** architecture to maximize speed.

1.  **Use All Available RAM**: The system aggressively utilizes available physical RAM to keep matrices in memory for maximum throughput.
2.  **Automatic Disk Spillover**: When physical RAM is exhausted, the system seamlessly switches to disk-backed storage (memory-mapped files).
3.  **Hybrid Async Pipeline**:
    *   **Small Matrices**: If a matrix fits entirely in VRAM, it is uploaded once and processed at maximum speed.
    *   **Large Matrices**: If a matrix exceeds VRAM (or the configured memory_limit), the system automatically switches to **Streaming Mode**.
        *   **CPU Parallelism**: The CPU uses multi-threading to pack data into pinned memory buffers.
        *   **GPU Overlap**: The GPU computes the current chunk while the CPU prepares the next one.

## 4. Configuration

While PyCauset works out-of-the-box, power users can fine-tune the behavior through the ComputeContext (exposed via `pycauset.cuda` for backward compatibility):

```python
import pycauset.cuda

# Enable with specific settings
pycauset.cuda.enable(
    device_id=0,              # Select specific GPU
    memory_limit=4*1024**3,   # Limit VRAM usage to 4GB
    enable_async=True         # Enable/Disable Async Pipelining
)
```

## 5. Troubleshooting

If you believe you have a GPU but PyCauset is using the CPU:

1.  Check if `pycauset.cuda.is_available()` returns `True`.
2.  Ensure `pycauset_cuda.dll` (Windows) or `libpycauset_cuda.so` (Linux) exists in the installation folder.
3.  Verify your NVIDIA drivers are installed and working (run `nvidia-smi`).
