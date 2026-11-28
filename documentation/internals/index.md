# Internals

How PyCauset is built and how to work on it. For contributors and for users who want to know
what happens under a call.

## Start here

1. `CLAUDE.md` (repository root): the short version and the rules that are easy to break;
   `review/README.md`, the ongoing review (plan, findings, log).
2. **[[internals/Architecture|Architecture]]**: the map of the moving parts and how a call
   travels through them.
3. **[[internals/Codebase Structure|Codebase Structure]]**: what lives where.
4. **[[project/protocols/Adding Operations|Adding Operations]]** and
   **[[project/protocols/Adding Types|Adding Types]]**: the procedures for changing the
   engine.

## How it works

*   **[[internals/Compute Architecture|Compute Architecture]]**: how an operation reaches a CPU kernel: `ComputeContext`, the kernel files, LAPACK, parallelism.
*   **[[internals/DType System|DType System]]**: scalar kinds, promotion rules, complex representation, overflow.
*   **[[internals/Memory and Data|Memory & Data]]**: tiered storage, the Memory Governor, the IO Accelerator, the object hierarchy.
*   **[[internals/Storage Semantics|Storage Semantics]]** and **[[internals/PyCauset Container Format|Container Format]]**: copy-on-write storage and the `.pycauset` file.
*   **[[internals/Out of Core|Out of Core]]**: out-of-core tiling.
*   **[[internals/Algorithms|Algorithms]]**: the solvers and how they are called.
*   **[[internals/Block Matrices|Block Matrices]]**: the Python-level block matrices.
*   **[[internals/Python Internals|Python Internals]]** and **[[internals/Bindings & Dispatch|Bindings & Dispatch]]**: the Python layer and the pybind11 bindings.
*   **[[internals/Square-only Assumptions|Square-only Assumptions]]**: where square shapes are required.

## Working on it

*   **[[internals/Build System|Build System]]**, **[[internals/Testing & Benchmarks|Testing & Benchmarks]]**.
*   **[[internals/Warnings & Exceptions|Warnings & Exceptions]]**: when the engine warns and when it raises.
*   **[[internals/Repository Hygiene|Repository Hygiene]]**, **[[internals/MACOS_WHEEL_PORTABILITY|macOS wheel portability]]**.
*   **[[project/protocols/Release Process|Release Process]]**.
