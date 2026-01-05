# R1_SAFETY: Robustness & Safety (The Shield)

## 1. The Problem
"The Happy Path is a lie." We need to ensure PyCauset survives crashes, power outages, and bad hardware states.
This plan focuses on **defensive engineering**: assuming the disk is slow, the GPU is broken, and the power will fail.

## 2. Phased Implementation Plan

### Phase 1: Storage Integrity (The Header)
**Goal:** Prevent data corruption when file formats evolve.
*   **Context:** Currently, .pycauset files are raw binary dumps. If we change the layout in R2, R1 readers will read garbage (or crash).
*   **Deliverables:**
    1.  **Header Definition:** Define a 64-byte struct at the start of every file.
        *   Magic: PYCAUSET (8 bytes)
        *   Version: uint32_t (4 bytes) - Set to 1.
        *   Reserved: 52 bytes of padding (zeroed) for future flags/checksums.
    2.  **Writer Update:** MemoryMapper writes this header on file creation.
    3.  **Reader Validation:** MemoryMapper reads and validates this header on open. Throw InvalidFileFormat if magic mismatches or version > supported.
    4.  **Offset Adjustment:** All payload offsets (matrix data) must shift by +64 bytes.

### Phase 2: Resource Management (The Leak)
**Goal:** Prevent "Ghost RAM" usage on Windows.
*   **Context:** IOAccelerator::discard() uses madvise(MADV_DONTNEED) on Linux, which frees physical pages. On Windows, it currently does nothing. Large temporary matrices consume page file/RAM even after we are "done" with them, leading to OOM on long runs.
*   **Deliverables:**
    1.  **Windows Implementation:** Implement discard_impl using VirtualUnlock (to unlock working set) and OfferVirtualMemory (if available) or VirtualFree(MEM_DECOMMIT) (if we can re-commit transparently, otherwise VirtualUnlock is the safest first step).
    2.  **Verification:** Create a test script that allocates 2x RAM size in chunks, discarding each after use, to prove we don't OOM.

### Phase 3: Compute Resilience (The Fallback)
**Goal:** Survive GPU instability.
*   **Context:** GPU drivers can crash or run out of memory. Currently, AutoSolver might retry or crash the process.
*   **Deliverables:**
    1.  **Pessimistic Initialization:** Wrap GPU context creation in 	ry/catch.
    2.  **Circuit Breaker:** If a GPU operation throws a hardware exception (OOM, CUDA error), catch it.
    3.  **Fallback Logic:**
        *   Log a warning: "GPU failed (error code). Falling back to CPU for remainder of session."
        *   Set gpu_device_ = nullptr.
        *   Retry the failed operation on CPU.
    4.  **Unit Test:** Mock a GPU failure (or force a bad allocation) and verify the system recovers.

### Phase 4: Data Persistence (The Flush)
**Goal:** Minimize data loss on power failure.
*   **Context:** OS file buffers are lazy. A power cut after "saving" might leave a file with zeroed pages.
*   **Deliverables:**
    1.  **Flush API:** Expose a robust lush() method in MemoryMapper.
        *   Windows: FlushFileBuffers.
        *   Linux: msync(MS_SYNC).
    2.  **Critical Path Integration:** Call lush() immediately after writing critical metadata (e.g., updating the "valid" flag of a matrix).
    3.  **Audit:** Review PersistentObject to ensure we don't have "torn write" windows where the file is structurally invalid.

### Phase 5: Hygiene (The Cleanup)
**Goal:** Keep the user's disk clean.
*   **Context:** If PyCauset crashes, it leaves pycauset_*.tmp files in .pycauset/. These accumulate forever.
*   **Deliverables:**
    1.  **Startup Scan:** On import pycauset, scan the .pycauset directory.
    2.  **Stale Detection:** Identify .tmp files that are not locked by any running process.
        *   *Note:* This is tricky on Linux (flock). On Windows, DeleteFile fails if open.
        *   *Strategy:* Try to delete. If it fails (locked), ignore. If it succeeds, good.
    3.  **Implementation:** Add clean_stale_files() to pycauset/__init__.py or src/core.

### Phase 6: Verification (The Gauntlet)
**Goal:** Prove robustness under fire and ensure no regressions.
*   **Context:** Safety features are hard to test because they handle rare events. We must simulate these events.
*   **Deliverables:**
    1.  **C++ Unit Tests:**
        *   	est_header_validation: Create files with bad magic/version and assert failure.
        *   	est_flush_behavior: Verify lush() calls succeed (mocking OS calls if necessary).
    2.  **Python Integration Tests:**
        *   	est_corrupt_load.py: Corrupt a .pycauset file header and assert pc.load() raises cleanly.
        *   	est_oom_resilience.py: Run the "Torture Test" (Phase 2) in CI (scaled to runner RAM).
    3.  **Benchmarks:**
        *   Run enchmarks/benchmark_io_smoke.py to ensure header/flush overhead is negligible.
        *   Verify AutoSolver fallback latency is acceptable.

### Phase 7: Documentation (The Manual)
**Goal:** Make safety features visible and explain the new guarantees.
*   **Context:** Users need to know about crash consistency and fallback behaviors. Contributors need to understand the file format.
*   **Deliverables:**
    1.  **Internals Update:**
        *   Update documentation/internals/MemoryArchitecture.md to document the .pycauset file header format (Magic + Version).
        *   Update documentation/internals/Compute Architecture.md to explain the GPU Circuit Breaker and Fallback logic.
    2.  **Guides Update:**
        *   Update documentation/guides/Storage and Memory.md to mention crash consistency guarantees and the .pycauset format versioning.
    3.  **Dev Handbook:**
        *   Update documentation/dev/Testing & Benchmarks.md with the new safety test patterns.

## 3. Success Criteria
*   [ ] **Phase 1:** Old files fail gracefully; new files load correctly.
*   [ ] **Phase 2:** "Torture test" (200GB alloc/free loop) passes on 16GB RAM laptop.
*   [ ] **Phase 3:** Simulated CUDA exception triggers CPU fallback without crashing Python.
*   [ ] **Phase 4:** Code audit confirms FlushFileBuffers usage.
*   [ ] **Phase 5:** .tmp files from killed processes disappear on next run.
*   [ ] **Phase 6:** All new tests pass; benchmarks show <5% regression on IO throughput.
*   [ ] **Phase 7:** All documentation updated per protocol.
