# Internals

This section documents the internal architecture and design decisions of PyCauset. It is intended for contributors and advanced users who want to understand how the library works under the hood.

## Architecture

*   **[[internals/Architecture|Architecture]]**: the map of the moving parts and how a call travels through them. Start here.
*   **[[internals/Compute Architecture|Compute Architecture]]**: How an operation reaches a CPU kernel: `ComputeContext`, `CpuDevice`, the kernel files, LAPACK, parallelism.
*   **[[internals/Memory and Data|Memory & Data]]**: Tiered storage, the Memory Governor, the IO Accelerator, the `.pycauset` format, and the object hierarchy.
*   **[[internals/Streaming Manager|Streaming Manager]]**: Shared policy for streaming/direct routing, tiling, queue depths, and IO observability.
*   **[[internals/Algorithms|Algorithms]]**: Details of the solvers (Eigenvalue, Matrix Multiplication) and their implementations.
*   **[[internals/DType System|DType System]]**: Scalar kinds (`bit`/`int`/`float`), promotion rules, complex representation, and overflow behavior.

## Process

*   **[[project/protocols/Release Process|Release Process]]**: Steps for releasing a new version of PyCauset.
