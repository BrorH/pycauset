# R1_SAFETY: Robustness & Safety (The Shield)

## 1. The Problem
"The Happy Path is a lie." We need to ensure PyCauset survives crashes, power outages, and bad hardware states.

## 2. Objectives

### 2.1 File Version Header (Storage)
*   **Risk:** R2 changes struct layout. R1 files crash R2 readers (or vice versa).
*   **Action:**
    *   Add a 64-byte header to `.pycauset` files.
    *   Magic Bytes: `PYCAUSET`
    *   Version: `uint32_t` (1)
    *   Check on load.

### 2.2 AutoSolver Pessimistic Fallback
*   **Risk:** GPU driver crashes or runs out of memory. `AutoSolver` currently retries or defaults to "Optimistic" (try GPU again).
*   **Action:**
    *   If GPU init fails, mark GPU as `BROKEN` for the session.
    *   Fallback to CPU immediately.
    *   Log a warning.

### 2.3 Crash Consistency
*   **Risk:** Power fail during write leaves file in corrupt state.
*   **Action:**
    *   Ensure `FlushFileBuffers` (Windows) / `msync` (Linux) is called after critical metadata writes.
    *   Verify atomicity of metadata updates (e.g., write new metadata to side buffer, then atomic switch? Or just accept "torn write" risk for R1 but minimize window).

### 2.4 Windows I/O Leak
*   **Risk:** `discard()` is a no-op on Windows. Large temp matrices consume Page File / RAM even after "freeing".
*   **Action:**
    *   Implement `VirtualUnlock` or `OfferVirtualMemory` in `IOAccelerator::discard`.

## 3. Deliverables
*   [ ] **File Header:** Implemented and verified.
*   [ ] **AutoSolver Safety:** Unit test simulating GPU failure.
*   [ ] **Crash Consistency:** Audit report.
*   [ ] **Windows Discard:** Implemented and verified.
