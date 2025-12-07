# Performance and Precision Guide

PyCauset is designed to automatically optimize performance based on your hardware. This guide explains how precision (Float32 vs Float64) is handled and how you can control it.

## Automatic Hardware Detection

When you load the CUDA accelerator, PyCauset queries your GPU's Compute Capability to determine the optimal floating-point precision.

*   **Consumer GPUs (GeForce GTX/RTX)**: These cards typically have excellent single-precision (Float32) performance but poor double-precision (Float64) performance (often 1/32 or 1/64 rate). PyCauset defaults to **Float32** on these systems.
*   **Data Center GPUs (Tesla P100/V100/A100)**: These cards have dedicated hardware for double-precision. PyCauset may default to **Float64** (or respect your input type more strictly) on these systems.

## Default Behavior

When you convert a NumPy array to a PyCauset matrix using `pycauset.asarray()`:

1.  **Float32 Input**: Always creates a `Float32Matrix`.
2.  **Float64 Input (Standard NumPy)**:
    *   If your hardware prefers Float32 (e.g., GTX 1060), PyCauset will **automatically downcast** to `Float32Matrix` for a 15-30x speedup.
    *   If your hardware supports fast Float64, it creates a `Float64Matrix`.

## Forcing Precision

If you require specific precision regardless of performance, you can force it by casting your NumPy array before passing it to PyCauset.

### Force Float64 (High Precision)
Use this if you need 64-bit accuracy, even if it runs slowly on your GPU.

```python
import numpy as np
import pycauset

# Create float64 array
a_np = np.random.rand(1000, 1000).astype(np.float64)

# PyCauset might downcast this by default on consumer GPUs.
# To force Float64, you currently need to use the specific class constructor (if exposed) 
# or ensure the auto-detection is disabled (feature coming soon).
# Currently, asarray() performs the optimization.
```

### Force Float32 (High Performance)
Use this to ensure maximum speed.

```python
import numpy as np
import pycauset

# Explicitly cast to float32
a_np = np.random.rand(1000, 1000).astype(np.float32)
a = pycauset.asarray(a_np) # Always creates Float32Matrix
```

## Benchmarks

On a GTX 1060 (Pascal):
*   **Float64**: ~135 GFLOPS
*   **Float32**: ~4400 GFLOPS (**32x faster**)

PyCauset ensures you get this speedup by default.

## Parallel Solvers

PyCauset implements a hybrid CPU/GPU strategy for complex linear algebra operations to ensure "it just works" regardless of your hardware or matrix size.

### Matrix Inversion & Eigenvalues

*   **GPU Acceleration**: If a CUDA-capable GPU is detected, operations like `inverse()` and `eigvals()` (dense) use NVIDIA's `cuSolver` library. This provides massive parallelism for $O(N^3)$ operations.
*   **Automatic Fallback**: GPU memory (VRAM) is limited. If your matrix is too large for the GPU, or if the operation fails for any reason, PyCauset **automatically falls back** to the multi-threaded CPU implementation. You do not need to write try-catch blocks or check memory manually.
*   **CPU Parallelism**: The CPU fallback uses OpenMP to parallelize block operations across all available cores.

| Operation | GPU Implementation | CPU Implementation |
| :--- | :--- | :--- |
| **Matrix Multiply** | `cuBLAS` (Async Pipelined) | OpenMP Blocked |
| **Inversion** | `cuSolver` (LU + Getri) | Block Gauss-Jordan (Parallel) |
| **Eigenvalues (Dense)** | `cuSolver` (QR/GeeV) | Sequential QR (w/ Parallel Reduction) |
| **Eigenvalues (Arnoldi)** | `cuBLAS` (Async Pipelined) | OpenMP (Mat-Vec) |


## Benchmark Results (Out-of-Core Solver)

The following benchmarks compare the performance of the Out-of-Core GPU solver against the CPU implementation on a consumer-grade GPU (NVIDIA GeForce GTX 1060).

### Double Precision (Float64)
Consumer GPUs often have reduced double-precision performance (1/32 rate).

| Size | CPU Time (s) | GPU Time (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 512 | 0.0200 | 0.0102 | **1.96x** |
| 1024 | 0.0762 | 0.0602 | **1.27x** |
| 2048 | 0.2497 | 0.3256 | 0.77x |
| 4096 | 0.9038 | 1.7921 | 0.50x |

*Note: For large matrices (N > 2048), the CPU's AVX2 instructions outperform the GPU's limited FP64 units.*

### Single Precision (Float32)
Consumer GPUs excel at single-precision arithmetic.

| Size | CPU Time (s) | GPU Time (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 512 | 0.0427 | 0.0127 | **3.35x** |
| 1024 | 0.0613 | 0.0145 | **4.24x** |
| 2048 | 0.1611 | 0.0459 | **3.51x** |
| 4096 | 0.5419 | 0.1929 | **2.81x** |

**Recommendation**: Use **Float32** for maximum performance on consumer hardware unless 64-bit precision is strictly required.

