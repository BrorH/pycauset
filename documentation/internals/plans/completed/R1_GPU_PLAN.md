# R1_GPU Plan: Robust, Cooperative Hardware Acceleration

**Status:** Completed
**Last updated:** 2026-02-08

## Executive Summary

The **R1_GPU** node turns the experimental CUDA backend into a robust, "Just Works" acceleration tier.

Crucially, we do **not** invent a new scheduler. We leverage the existing, sophisticated **`AsyncStreamer`** architecture. The goal is to build **Algorithm Drivers**—smart host-side loops that orchestrate the `AsyncStreamer` logic to handle complex operations (Inversion, Eigen) on datasets larger than GPU memory.

## Core Philosophy

1.  **"Just Works" (No Tuning Required)**: The system automatically detects hardware, benchmarks capabilities, and routes operations. Defaults are safe.
2.  **Streaming First**: We assume data does *not* fit in GPU RAM. All implementations must support tiling via the `MemoryGovernor`.
3.  **Cooperative Pipelining**: We rarely split a single math op between CPU/GPU (bandwidth contention). Instead, we pipeline: CPU handles logistics (prefetch/pin/format) in parallel with GPU compute.
4.  **Plug-and-Play Routines**: Each operation’s GPU acceleration path must be added through a stable, documented contract so new ops do **not** require deep integration knowledge.
5.  **CPU is First-Class**: GPU work must be co-designed with the future R1_CPU pipeline. Drivers, routing, and streaming contracts must be backend-agnostic so CPU and GPU can share the same orchestration model.

---

## Deliverables & Phases

**Note**: This plan is **not** documentation. Each phase must end with real doc updates that follow the Documentation Protocol.

### Phase 1: Robust Discovery & "Just Works" Dispatch — DONE
**Goal**: Make the `AutoSolver` smart enough to trust by default.

*   **Hardware Audit & Persistence**:
    *   Query `cudaGetDeviceProperties`.
    *   **Micro-benchmark**: Run silent `SGEMM`/`DGEMM` startup test.
    *   **Cache**: Save results to `~/.pycauset/hardware_profile.json` to skip benchmarks on future runs.
    *   **Dynamic Pinning Budget**: Implement heuristic `Budget = min(SystemRAM * 0.5, FreeRAM * 0.8, 8GB)` to safely maximize I/O throughput.
*   **Cost Model Dispatch**:
    *   Replace fixed size thresholds with a transfer-vs-compute cost equation:
        $$ T_{gpu} = \frac{\text{Bytes}}{\text{BW}_{pci}} + \frac{\text{Ops}}{\text{FLOPS}_{gpu}} + T_{latency} $$
*   **Control Surface (Python)**:
    *   `pycauset.cuda.force_backend(...)`
    *   `pycauset.cuda.set_pinning_budget(bytes)`
    *   `pycauset.cuda.benchmark(force=True)`
*   **Documentation Checkpoint (required, per [[project/protocols/Documentation Protocol.md|Documentation Protocol]])**:
    *   [x] Update `internals/Compute Architecture.md`.
    *   [x] Add `docs/functions/pycauset.cuda.*` API references.
    *   [x] Add `docs/parameters/pinning_budget.md`.

### Phase 2: Streaming Algorithm Drivers (Host-Orchestrated) — DONE
**Goal**: Implement robust out-of-core drivers for complex linear algebra by orchestrating the existing `AsyncStreamer`.

*   **Strategy: Host-Side Orchestration**:
    *   We reject a generic runtime scheduler.
    *   We implement **Algorithm-Specific Drivers** (C++ classes) where the CPU executes the algorithm state machine (loops, barriers, dependency checks).
    *   The Host submits batches of independent tiles to `AsyncStreamer`, which handles the low-level double-buffered prefetch/pin/issue pipeline.
*   **Deliverables (The "Drivers")**:
    *   **`MatmulDriver`**: Verification of existing flat streaming for 100GB+ scale.
    *   **`CholeskyDriver`**: Implement Right-Looking Block Cholesky. Host logic manages dependencies (Diagonal wait -> Panel update -> Trailing update).
    *   **`ArnoldiDriver`**: Implement Block Arnoldi. Optimizes memory bandwidth by fusing multiple vector updates into blocked operations.
*   **Plug-and-Play Routine Contract (required)**:
    *   Define a **driver/routine contract** that every new GPU-accelerated op must follow (inputs, tiling semantics, error handling, fallback, and registration).
    *   Require **capability declarations** (supported dtypes/structures/properties) so `AutoSolver` routing is explicit and testable.
    *   Require **CPU compatibility**: the same contract must be usable by the planned R1_CPU worker pipeline (no GPU-only hidden paths).
    *   Require **debug trace tags** for routing, I/O, and kernel execution to keep diagnostics deterministic.
*   **Documentation Checkpoint (required, per [[project/protocols/Documentation Protocol.md|Documentation Protocol]])**:
    *   [x] Update `internals/Streaming Manager.md` to document the Driver pattern and the driver/routine contract.
    *   [x] Update `project/protocols/Adding Operations.md` with a **GPU routine authoring checklist** (plug-and-play steps).
    *   [x] Update `guides/Performance Guide.md` with a section on "Streaming Constraints" and CPU/GPU orchestration caveats.

### Phase 3: Integration (Properties & Heterogeneity) — DONE
**Goal**: Routing respects semantic properties to exploit PyCauset's structural advantage.

*   **Mechanism: Traits-Based Dispatch (Tag Dispatch)**:
    *   Implement C++20 `MatrixTraits` derived from **R1_PROPERTIES**.
    *   `AutoSolver` selects kernels based on `BestKernel(Op, Traits)`, decoupling policy from execution.
*   **Feature: C++ Property Mirroring (Fast Flags)**:
    *   Since properties are currently Python-only (see `Storage and Memory.md` -> "Technical Implementation"), `AutoSolver` cannot currently see them.
    *   Add a `uint64_t properties_flags` field to `PersistentObject` in C++.
    *   Wire the Python setter to update this C++ bitmask in $O(1)$.
*   **The "PyCauset Advantage"**:
    *   We use physics knowledge (Causality=Triangular, Propagator=Symmetric) to select $O(N^2)$ shortcuts.
*   **Documentation Checkpoint (required, per [[project/protocols/Documentation Protocol.md|Documentation Protocol]])**:
    *   [x] Update key `docs/classes/Matrix.md` regarding how properties affect performance.
    *   [x] Update `internals/algorithms.md` with property-specific complexity guarantees.
    *   [x] Update `project/protocols/Adding Operations.md` to include steps for registering new Traits/Tags.

### Phase 4: Block Matrix Orchestration — DONE
**Goal**: Ensure the `BlockMatrix` composite structure utilizes the new drivers.

*   **Routing**: Ensure `BlockMatrix` operations (which decompose into sub-ops) route those sub-ops through `AutoSolver` correctly.
*   **Heterogeneity**: Handle cases where one block is on GPU and another must stay on CPU (utilizing the unified worker interface defined in **R1_CPU**).
*   **Documentation Checkpoint (required, per [[project/protocols/Documentation Protocol.md|Documentation Protocol]])**:
    *   [x] Update `internals/Block Matrices.md` with GPU/CPU routing notes for block ops.

### Phase 5: Robustness, Documents & Benchmarking (The "Shield")
**Goal**: Final polish, stress testing, and complete documentation.

*   **Stress Test**: Run the "100GB Matrix" test case on a machine with <16GB RAM and <8GB VRAM. **[DONE]**
*   **Unit Tests**: Mock `cudaGetDeviceProperties` to simulate hardware states. **[DONE]**
*   **Documentation Sweep (required, per [[project/protocols/Documentation Protocol.md|Documentation Protocol]])**:
    *   **Guide**: Add "GPU Configuration" section to existing `guides/Performance Guide.md`. **[DONE]**
    *   **Internal**: Update `documentation/project/protocols/Adding Operations.md` with GPU-specific kernel checklist. **[DONE]**
    *   **Compliance**: Verify all new APIs have `docs/` entries. **[DONE]**

---

## Technical Constraints & Decisions

### 1. The PCI-e Bottleneck
Relying on the GPU for everything is often slower for medium sizes due to transfer overhead.
*   **Decision**: The "Cost Model" (Phase 1) is strict. If the model says CPU is faster, we must use CPU.

### 2. Pinned Memory Scarcity
Pinned memory locks physical RAM.
*   **Decision**: We use the `MemoryGovernor`'s dynamic pinning budget. Drivers must request a "ticket" to pin. If denied, they degrade to pageable memory (slower) but do not crash.

### 3. Precision & Properties
GPU paths must respect **R1_PROPERTIES**. Specialized kernels (Syrk, Trmm) are preferred over generic Gemm to save 50% FLOPs.

## Open Questions

*   **Context Management**: Global pool vs Per-Thread? (Likely global pool for R1).
*   **Multi-GPU**: Explicitly out-of-scope for R1.