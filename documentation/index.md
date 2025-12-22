# Welcome to PyCauset

**PyCauset** is a high-performance Python library for numerical Causal Set Theory. It is designed to bridge the gap between abstract mathematical models and large-scale numerical simulations.

## The Philosophy: Tiered Storage

Causal sets are computationally demanding. For a set of size $N$, the causal matrix is $O(N^2)$. For $N=100,000$, a dense matrix requires gigabytes of memory.

PyCauset solves this with a **Hybrid Architecture**:
1.  **RAM-First**: Small matrices behave exactly like NumPy arrays.
2.  **Disk-Backed**: Large matrices automatically spill to memory-mapped files (single-file `.pycauset` containers).
3.  **Bit-Packing**: Causal relations are stored as single bits, reducing memory usage by 64x compared to standard integers.

## Documentation Structure

### 📘 [[guides/index|User Guides]]
Practical tutorials and conceptual explanations.
*   **[[guides/Installation|Installation]]**: Install PyCauset.
*   **[[guides/User Guide|User Guide]]**: First steps and core workflow.
*   **[[guides/Causal Sets|Causal Sets]]**: Working with the core `CausalSet` object.
*   **[[guides/Field Theory|Field Theory]]**: Simulating quantum fields and propagators.
*   **[[guides/Visualization|Visualization]]**: Interactive 3D plotting.
*   **[[guides/Performance Guide|Performance]]**: GPU acceleration and precision tuning.
*   **[[guides/Storage and Memory|Storage]]**: Understanding the file formats and memory management.

### ⚙️ [[docs/index|API Reference]]
Detailed documentation of classes and functions.
*   **[[docs/classes/index|Classes]]**: `CausalSet`, `Matrix`, `Vector`, `Spacetime`.
*   **[[docs/functions/index|Functions]]**: `matmul`, `inverse`.

### 🧠 [[internals/index|Internals]]
Deep dive into the C++ core for contributors.
*   **[[internals/Compute Architecture|Compute Architecture]]**: CPU/GPU dispatch and solvers.
*   **[[internals/MemoryArchitecture|Memory Architecture]]**: Tiered storage, Governor, and CoW.
*   **[[internals/Memory and Data|Memory & Data]]**: The `.pycauset` container format and memory mapping.
*   **[[internals/Algorithms|Algorithms]]**: Mathematical derivations and implementation details.

### 🚀 [[project/index|Project]]
*   **[[project/Philosophy|Philosophy]]**: Design mantras.
*   **[[project/Contributing|Contributing]]**: How to build and test.
*   **[[TODO|Roadmap]]**: Future plans.

### 🧰 [[dev/index|Dev Handbook]]
High-signal onboarding for contributors.
*   **[[dev/Restructure Plan|Restructure Plan]]**: The approved reorganization plan and gates.

## Citation

If you use PyCauset in your research, please cite the repository:
[https://github.com/BrorH/pycauset](https://github.com/BrorH/pycauset)
