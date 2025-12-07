# GPU Acceleration

PyCauset includes a high-performance GPU backend powered by NVIDIA CUDA. This backend is designed to be **frictionless**: it requires no configuration, automatically detects compatible hardware, and seamlessly falls back to the CPU if no GPU is available.

## Features

*   **Automatic Detection**: The library checks for a CUDA-capable GPU at startup.
*   **Dynamic Loading**: The GPU backend is a separate plugin (`pycauset_cuda.dll` / `.so`). You do not need CUDA installed to run PyCauset on a CPU-only machine.
*   **Out-of-Core Processing**: Algorithms are designed to handle matrices larger than your GPU's VRAM by streaming data efficiently between Disk, RAM, and VRAM.

## Requirements

To use the GPU acceleration, you need:

1.  **Hardware**: An NVIDIA GPU with Compute Capability 6.0 or higher (Pascal or newer).
2.  **Drivers**: Up-to-date NVIDIA Drivers.
3.  **Software**: The `pycauset_cuda` plugin must be present in the library directory (installed automatically if built with CUDA support).

## Supported Operations

The following operations are currently accelerated:

*   **Eigenvalue Decomposition** (`eigvals`, `eigvals_arnoldi`, `eigvals_skew`):
    *   Uses `cuSOLVER` for dense matrices.
    *   Uses a custom streaming kernel for the Arnoldi/Lanczos iterations, allowing for massive scale spectral analysis.
*   **Matrix Multiplication** (`matmul`, `*` operator):
    *   Uses `cuBLAS` for high-performance GEMM.
    *   **Note on BitMatrices**: `TriangularBitMatrix` multiplication is **NOT** accelerated on GPU by default. The overhead of converting packed bits to floats outweighs the compute benefits for most cases. To use the GPU, you must explicitly convert your matrices: `A.to_float() @ B.to_float()`.
*   **Matrix Inversion** (`inverse`):
    *   Uses `cuSOLVER` LU factorization.

## Configuration

While PyCauset works out-of-the-box, power users can fine-tune the behavior:

```python
import pycauset.cuda

# Enable with specific settings
pycauset.cuda.enable(
    device_id=0,              # Select specific GPU
    memory_limit=4*1024**3,   # Limit VRAM usage to 4GB
    enable_async=True         # Enable/Disable Async Pipelining
)
```

## Performance & Memory

PyCauset uses a **Hybrid Async Pipeline** to maximize performance:

*   **Small Matrices**: If a matrix fits entirely in VRAM, it is uploaded once and processed at maximum speed.
*   **Large Matrices**: If a matrix exceeds VRAM (or the configured `memory_limit`), the system automatically switches to **Streaming Mode**.
    *   **CPU Parallelism**: The CPU uses multi-threading to pack data into pinned memory buffers.
    *   **GPU Overlap**: The GPU computes the current chunk while the CPU prepares the next one.
    *   **Result**: You can process matrices of virtually any size (limited only by system RAM) with performance close to the theoretical hardware limit.
*   **Large Matrices**: If a matrix exceeds VRAM, PyCauset uses **Asynchronous Streaming**.
    *   **Pipelining**: The CPU loads the next chunk of data while the GPU computes the current chunk.
    *   **Pinned Memory**: Uses specialized OS memory for faster transfer speeds.
    *   This allows you to process 100GB+ matrices on an 8GB GPU, limited only by PCIe bandwidth.

## Limitations

*   **Matrix Inversion**: Currently, `inverse()` requires the entire matrix to fit in VRAM. If the matrix is too large, it will fall back to CPU or raise an error.
*   **BitMatrices**: As noted above, BitMatrices are processed on the CPU by default.

## Troubleshooting

If you believe you have a GPU but PyCauset is using the CPU:

1.  Check if `pycauset.cuda.is_available()` returns `True`.
2.  Ensure `pycauset_cuda.dll` (Windows) or `libpycauset_cuda.so` (Linux) exists in the installation folder.
3.  Verify your NVIDIA drivers are installed and working (run `nvidia-smi`).
