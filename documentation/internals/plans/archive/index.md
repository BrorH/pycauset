# Plan Archive (R1 and earlier)

This directory holds the **historical plan artifacts** from Release 1 (v0.6.x) and earlier. They
are archived for reference only, they are **not** the current plan of record, and they are **not**
rendered in the docs navigation.

For **what's happening now**, see:

- Forward roadmap: [`../TODO.md`](../TODO.md) ("Roadmap" in the docs nav)
- R1 release record: [`../R1_EXECUTION.md`](../R1_EXECUTION.md)
- Deferred optimization map (now the R2 engine track): [`../OPTIMIZATION_STATUS.md`](../OPTIMIZATION_STATUS.md)
- Support-readiness framework (active): [`../SUPPORT_READINESS_FRAMEWORK.md`](../SUPPORT_READINESS_FRAMEWORK.md)
- R2 plans (active): [`../../../project/plans/`](../../../project/plans/)

> **Consolidation note (R2 start):** the `plans/completed/` directory and several stray `R1_*`
> plan files were merged into this single archive to stop plan-file sprawl. Where duplicated copies
> existed, the more current revision was kept here and the rest removed.

## Contents

| File | What it was | Disposition |
| :-- | :-- | :-- |
| `BLAS_INTEGRATION_PLAN.md` | OpenBLAS integration for dense GEMM | Done, superseded by `OPTIMIZATION_STATUS.md` |
| `DTYPE_COMPLEX_OVERFLOW_PLAN.md` | dtype/complex/overflow policy + complex-float integration | Complete |
| `phase1_inventory.md` | R1_NUMPY interop inventory report | Complete |
| `R1_BLOCKMATRIX_PLAN.md` | Block matrices + heterogeneous dtypes | Complete |
| `R1_CPU_PLAN.md` | Modern tiled CPU engine | Deferred → R2 engine track (`R2_CPU` / `R2_STREAM`) |
| `R1_GPU_PLAN.md` | GPU routing + parity | Routing done; parity → `R2_GPU` |
| `R1_IO_PLAN.md` | Out-of-core I/O + NumPy interop surface | Complete |
| `R1_LAZY.md` | Lazy evaluation + RAM-first persistence | Complete |
| `R1_LINALG_PLAN.md` | Linalg surface completeness | Complete |
| `R1_NUMPY_PLAN.md` | Fast NumPy interop | Complete |
| `R1_PERF.md` | R1 performance optimizations (threading, I/O, AVX-512) | Complete |
| `R1_POLISH.md` | Professionalism / polish checklist | Remaining items tracked in `../TODO.md` |
| `R1_PROPERTIES_PLAN.md` | Semantic properties + property-aware algebra | Complete |
| `R1_SAFETY.md` | Robustness & safety | Complete |
| `R1_SHAPES_PLAN.md` | NxM shape support | Complete |
| `R1_STORAGE_PLAN.md` | Single-file `.pycauset` container | Complete |
| `Restructure Plan.md` | Codebase restructure execution record | Complete |
