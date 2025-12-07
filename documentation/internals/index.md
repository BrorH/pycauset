# Internals

This section documents the internal architecture and design decisions of PyCauset. It is intended for contributors and advanced users who want to understand how the library works under the hood.

## Architecture

*   **[[ComputeSystem]]**: The unified CPU/GPU compute architecture, including `ComputeContext`, `AutoSolver`, and parallelization strategies.
*   **[[MemorySystem]]**: Memory management, file-backed storage, and the `.pycauset` file format.
*   **[[Matrix and Vector Architecture]]**: The C++ class hierarchy for matrix and vector types.
*   **[[Stateless Sprinkling]]**: How coordinates are regenerated on demand.

## Algorithms

*   **[[Algorithms]]**: Details of the solvers (Eigenvalue, Matrix Multiplication) and their implementations.
*   **[[Math Derivation]]**: Mathematical background for the implemented algorithms.

## Process

*   **[[Release Process]]**: Steps for releasing a new version of PyCauset.
