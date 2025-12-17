# API Reference

This section contains the detailed API documentation for the PyCauset library.

## Core Modules

*   **[[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]]**: The main class representing a causal set.
*   **[[docs/classes/spacetime/pycauset.spacetime.md|pycauset.spacetime]]**: Spacetime manifolds and geometry.
*   **[[docs/pycauset.vis/index.md|pycauset.vis]]**: Visualization tools.
*   **[[docs/classes/field/index.md|pycauset.field]]**: Quantum field helpers.

## Data Structures

*   **[[docs/classes/matrix/index.md|Matrix classes]]**: Dense, triangular, bit-packed, and structured matrices.
*   **[[docs/classes/vector/index.md|Vector classes]]**: Disk-backed and specialized vectors.

## Functions

*   **[[docs/functions/pycauset.matrix.md|pycauset.matrix]]** / **[[docs/functions/pycauset.vector.md|pycauset.vector]]**: Construct from data.
*   **[[docs/functions/pycauset.zeros.md|pycauset.zeros]]** / **[[docs/functions/pycauset.ones.md|pycauset.ones]]** / **[[docs/functions/pycauset.empty.md|pycauset.empty]]**: Allocate with explicit `dtype`.
*   **[[docs/functions/pycauset.causal_matrix.md|pycauset.causal_matrix]]**: Create a causal matrix (triangular bit matrix).
*   **[[docs/functions/pycauset.causet.md|pycauset.causet]]**: Convenience constructor for [[pycauset.CausalSet]].
*   **[[docs/functions/pycauset.matmul.md|pycauset.matmul]]**: Matrix multiplication.
*   **[[docs/functions/pycauset.invert.md|pycauset.invert]]**: Matrix inversion.
*   **[[docs/functions/pycauset.load.md|pycauset.load]]** / **[[docs/functions/pycauset.save.md|pycauset.save]]**: Persistence.
*   **[[docs/functions/pycauset.compute_k.md|pycauset.compute_k]]**: Propagator-related helper.
