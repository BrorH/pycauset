# Stress Test Simulation Report

## Overview
This report simulates the installation and execution of `pycauset` on 30 diverse user systems. The goal is to identify potential failure points in installation, runtime, and performance under stress (e.g., 50,000 x 50,000 matrix operations).

**Test Protocol:**
1.  **Install**: `pip install pycauset` (simulated).
2.  **Stress Test**:
    *   Create 50k x 50k matrix (approx. 20GB for double precision).
    *   Perform inversion or large-scale multiplication.
    *   Run standard NumPy interop tests.

## System Findings

| ID | OS | CPU | RAM | GPU | Python | Outcome | Failure Analysis / Notes |
|----|----|-----|-----|-----|--------|---------|--------------------------|
| 1 | Windows 11 | i9-13900K | 64GB | RTX 4090 | 3.11 | **PASS** | Ideal scenario. CUDA loads, AVX2 supported. 64GB RAM handles 20GB matrix easily. |
| 2 | Windows 10 | i7-8700K | 16GB | GTX 1080 | 3.10 | **FAIL (Runtime)** | **OOM**. 50k x 50k double matrix requires ~18.6GB. System swaps heavily or crashes. |
| 3 | Windows 10 | Ryzen 5 3600 | 32GB | None | 3.9 | **PASS (Slow)** | CPU fallback works. 32GB RAM is sufficient but tight. Inversion will be very slow without GPU. |
| 4 | Ubuntu 22.04 | Threadripper | 128GB | A100 (80GB) | 3.10 | **PASS** | Excellent performance. Linux build needs to ensure `libpycauset_cuda.so` is found. |
| 5 | macOS 14 | M2 Max | 32GB | Integrated | 3.11 | **FAIL (Install/Runtime)** | **Architecture**. No ARM64 wheels provided? `CMakeLists.txt` has some Apple logic but OpenMP might be tricky. No CUDA support on Mac. |
| 6 | Windows 11 | i5-12400 | 8GB | RTX 3060 | 3.12 | **FAIL (Runtime)** | **OOM**. 8GB is insufficient for 50k x 50k matrix. Memory mapping might help but performance will be abysmal due to disk thrashing. |
| 7 | Linux (CentOS 7) | Xeon E5 v3 | 64GB | Tesla K80 | 3.8 | **FAIL (Runtime)** | **CUDA Compat**. K80 is old (Compute Capability 3.7). If compiled for newer arch (e.g., 7.0+), it will fail to load kernel. `glibc` version might also be an issue for wheels. |
| 8 | Windows 10 | Core 2 Duo | 4GB | None | 3.8 | **FAIL (Install/Runtime)** | **Instruction Set**. If compiled with AVX2 (default in MSVC often for modern builds), it will crash with "Illegal Instruction". 4GB RAM is non-starter. |
| 9 | Windows 11 | i7-11800H | 16GB | RTX 3050 Ti | 3.11 | **FAIL (Runtime)** | **OOM**. 16GB RAM shared with OS. 20GB allocation fails. |
| 10 | Ubuntu 20.04 | i7-9700K | 32GB | GTX 1660 | 3.9 | **PASS** | Works. GTX 1660 has no Tensor Cores but CUDA should work. 6GB VRAM is too small for 50k x 50k on GPU, must fallback to CPU or handle OOM gracefully. |
| 11 | Windows 11 | Surface Pro X (ARM) | 16GB | None | 3.11 | **FAIL (Install)** | **No Wheel**. Likely no Windows ARM64 wheel available. Source build requires ARM64 MSVC tools. |
| 12 | macOS 13 | i9 (Intel) | 64GB | AMD Pro 5500M | 3.10 | **PASS (CPU Only)** | Works on CPU. AMD GPU not supported (CUDA only). |
| 13 | Windows 10 | i5-6500 | 16GB | GT 710 | 3.9 | **PASS (CPU Only)** | GT 710 is Kepler (CC 3.5), likely dropped in modern CUDA. `ComputeContext` should fail to load CUDA and fallback to CPU silently. |
| 14 | Fedora 38 | Ryzen 9 7950X | 64GB | RTX 4080 | 3.11 | **PASS** | `libpycauset_cuda.so` loading depends on `LD_LIBRARY_PATH` or RPATH. If not set correctly, falls back to CPU. |
| 15 | Windows Server | Xeon Gold | 256GB | None | 3.10 | **PASS** | Pure CPU powerhouse. AVX-512 might be used if compiled with `/arch:AVX512` (check CMake). |
| 16 | Windows 11 | i3-10100 | 8GB | None | 3.12 | **FAIL (Runtime)** | **OOM**. |
| 17 | Arch Linux | i7-13700K | 32GB | Arc A770 | 3.11 | **PASS (CPU Only)** | Intel Arc not supported. Fallback to CPU. |
| 18 | Windows 10 | i7-4790K | 16GB | GTX 970 | 3.8 | **FAIL (Runtime)** | **OOM**. |
| 19 | Ubuntu 22.04 | AWS g4dn.xlarge | 16GB | T4 | 3.10 | **FAIL (Runtime)** | **OOM**. 16GB RAM is bottleneck. T4 has 16GB VRAM, also tight for 20GB matrix. |
| 20 | Windows 11 | i9-14900K | 128GB | 2x RTX 4090 | 3.11 | **PASS** | Multi-GPU not explicitly handled in `ComputeContext` (uses `device_id=0` default). Works on one GPU. |
| 21 | Windows 7 | i7-2600K | 16GB | GTX 580 | 3.8 | **FAIL (OS/Runtime)** | Python 3.9+ dropped Win7. Python 3.8 might work. GTX 580 (Fermi) definitely not supported by modern CUDA. |
| 22 | macOS 12 | M1 | 8GB | Integrated | 3.9 | **FAIL (Runtime)** | **OOM**. 8GB unified memory is insufficient. |
| 23 | Windows 10 | Ryzen 7 5800X | 32GB | RTX 3070 | 3.13 (Beta) | **FAIL (Install)** | **No Wheel**. Python 3.13 wheels likely not built yet. Requires source build (needs MSVC installed). |
| 24 | Ubuntu 24.04 | i5-13600K | 32GB | RTX 4060 | 3.11 | **PASS** | `glibc` version on 24.04 is new, wheels built on older manylinux should work. |
| 25 | Windows 11 | i7-12700H | 32GB | RTX A2000 | 3.10 | **PASS** | Workstation GPU works fine with CUDA. |
| 26 | Debian 11 | Atom C3000 | 4GB | None | 3.9 | **FAIL (Runtime)** | **Timeout/OOM**. Atom is too weak, 4GB RAM too low. |
| 27 | Windows 10 | i5-4590 | 8GB | GTX 750 Ti | 3.10 | **FAIL (Runtime)** | **OOM**. |
| 28 | Windows 11 | i7-10700 | 64GB | Radeon RX 6800 | 3.11 | **PASS (CPU Only)** | AMD GPU ignored. |
| 29 | Ubuntu 20.04 | EPYC 7763 | 512GB | A100 (40GB) | 3.8 | **PASS** | A100 40GB VRAM is enough for *some* operations, but 50k x 50k double is ~18.6GB. It fits! |
| 30 | Windows 11 | i5-13400F | 32GB | RTX 4070 | 3.11 | **PASS** | Good mid-range setup. |

## Critical Shortcomings Identified

1.  **Memory Management (OOM)**:
    *   **Issue**: A 50,000 x 50,000 matrix of `double` takes ~18.6 GB ($50000^2 \times 8$ bytes). Systems with 16GB or less RAM (very common) will crash or swap to death.
    *   **Fix Needed**: Implement out-of-core processing or tiled algorithms for large matrices. The `MemoryMapper` exists but `CpuSolver` seems to load data into pointers (`a_dense->data()`) which might trigger page faults, but standard algorithms (like `matmul_impl`) iterate in a way that might cause thrashing if not blocked correctly for disk I/O.

2.  **GPU Memory Limits**:
    *   **Issue**: Even high-end GPUs (RTX 3080/4070) often have 10-12GB VRAM. The 20GB matrix won't fit.
    *   **Fix Needed**: `ComputeContext` needs to handle VRAM oversubscription or fallback to CPU/System RAM (Unified Memory) explicitly. Currently, `create_cuda_device` might succeed, but allocation or kernel launch will fail.

3.  **CUDA Dependency & Loading**:
    *   **Issue**: `ComputeContext.cpp` tries to load `pycauset_cuda.dll` / `libpycauset_cuda.so`. If the user installs via pip, are these shared libraries bundled correctly?
    *   **Fix Needed**: Ensure wheels include the CUDA runtime (or rely on system CUDA) and that the DLL search path is correct.

4.  **Instruction Set Architecture (ISA)**:
    *   **Issue**: If compiled with AVX2 enabled (common for performance), older CPUs (pre-Haswell, Celerons, Atoms) will crash with "Illegal Instruction".
    *   **Fix Needed**: Runtime dispatch for AVX2/AVX-512 or distribute generic binaries.

5.  **macOS / ARM Support**:
    *   **Issue**: No mention of Metal (MPS) support. Mac users are stuck with CPU. ARM64 (M1/M2) requires specific build targets.
    *   **Fix Needed**: Add ARM64 CI builds. Consider Metal backend for Apple Silicon.

6.  **Missing Wheels**:
    *   **Issue**: Python 3.13, Windows ARM64, Linux aarch64 likely don't have pre-built wheels.
    *   **Fix Needed**: Expand CI matrix (cibuildwheel).

## Conclusion
"IT JUST WORKS" is currently **false** for:
*   Users with < 32GB RAM (for large stress tests).
*   Users with < 24GB VRAM (trying to use GPU for large matrices).
*   Mac users expecting acceleration.
*   Users with older CPUs if AVX is forced.

The system is robust for high-end workstations but fragile for consumer hardware when scaling up.
