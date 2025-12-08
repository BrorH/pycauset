# PyCauset Performance Benchmarks

This document summarizes the performance improvements achieved by the new memory management architecture (Tiered Storage, Copy-on-Write, and IO Accelerator).

## System Configuration
- **OS**: Windows
- **RAM**: ~32 GB (Estimated)
- **Disk**: NVMe/SSD

## Summary of Results

| Optimization | Benchmark | Baseline Time | Optimized Time | Speedup |
|--------------|-----------|---------------|----------------|---------|
| **Copy-on-Write (CoW)** | Matrix Copy (500MB) | 0.4200 s | 0.00003 s | **14,000x** |
| **Deferred Persistence** | Create/Destroy 5k Objects | 1.97 s | 1.87 s | **1.05x** |
| **IO Accelerator** | Read Scan (50GB) | 114.57 s | 63.54 s | **1.80x** |

---

## Detailed Analysis

### 1. Copy-on-Write (CoW)
*   **Scenario**: Duplicating a matrix for a hypothetical causal intervention.
*   **Result**: Instantaneous.
*   **Why**: Instead of copying 500MB of data, we simply increment a reference count. Data is only copied if modified.

### 2. Deferred Persistence (Tiered Storage)
*   **Scenario**: Creating 5,000 small temporary matrices (1KB each).
*   **Result**: 5% faster.
*   **Why**: Small objects are kept in RAM (`malloc`), avoiding the overhead of creating, locking, and deleting files on the NTFS filesystem. The OS is already fast at file creation, but RAM is faster.

### 3. IO Accelerator (Sliding Window)
*   **Scenario**: Reading a **50GB** dataset (larger than physical RAM) sequentially.
*   **Result**: **1.80x Faster** (almost double the speed).
*   **Why**: 
    *   **Baseline**: The OS tries to cache the file. When RAM fills up (32GB), the OS must frantically swap out old pages to make room for new ones. This "thrashing" slows down the read speed to ~450 MB/s.
    *   **Optimized**: The `IOAccelerator` explicitly tells the OS: *"I am done with this 64MB chunk, throw it away."* (`MADV_DONTNEED`). This keeps the memory footprint low, preventing the OS from ever hitting the "RAM Full" panic state. The read speed stays high (~800 MB/s).

## Conclusion
The new architecture successfully handles "Big Data" workloads.
- **Small Data**: Faster (due to RAM-first allocation).
- **Big Data**: Much Faster (due to active memory management).
- **Huge Data (>RAM)**: **Enabling** (prevents system freeze/thrashing).
