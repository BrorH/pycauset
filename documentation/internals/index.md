# Internals

This section documents the internal architecture and design decisions of PyCauset. It is intended for contributors and advanced users who want to understand how the library works under the hood.

## Architecture

*   **[[Compute Architecture]]**: The unified CPU/GPU compute architecture, including `ComputeContext`, `AutoSolver`, and parallelization strategies.
*   **[[MemoryArchitecture]]**: The Tiered Storage system, Memory Governor, IO Accelerator, and Copy-on-Write mechanism.
*   **[[Memory and Data]]**: The `.pycauset` file format and the Object Hierarchy.
*   **[[Algorithms]]**: Details of the solvers (Eigenvalue, Matrix Multiplication) and their implementations.

## Process

*   **[[Release Process]]**: Steps for releasing a new version of PyCauset.
