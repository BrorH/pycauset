# Protocol: Adding New Matrix and Vector Types

**Objective**: Ensure new matrix/vector structures are fully integrated into the factory, storage, compute, and Python ecosystems.

## Step-by-step implementation guide

### 1) Core definitions (`include/pycauset/core/Types.hpp`)

- Add a new enum value to `MatrixType`.
- Decide whether the new type is a matrix-like (`rows x cols`) or a vector-like (`n x 1`) object.

### 2) Define the class

- Matrix types live in `include/pycauset/matrix/`.
- Vector types live in `include/pycauset/vector/`.

Implement the required virtual interface:

- Matrices (`MatrixBase`): `get_element_as_double(i, j)`, `multiply_scalar`, `add_scalar`, `transpose`, and correct `clone()` behavior.
- Vectors (`VectorBase`): `get_element_as_double(i)`, `multiply_scalar`, `add_scalar`, `transpose` (if supported), and correct `clone()` behavior.

If the type is intended for GPU/BLAS interoperability, it must expose contiguous raw memory access where appropriate.

### 3) Update the factory (`include/pycauset/core/ObjectFactory.hpp`, `src/core/ObjectFactory.cpp`)

The factory is the central registry for creating, loading, and cloning persistent objects.

- Matrices:
    - `create_matrix`
    - `load_matrix`
    - `clone_matrix`
- Vectors:
    - `create_vector`
    - `load_vector`
    - `clone_vector`

### 4) Update compute support

- Add CPU support in `src/compute/cpu/CpuSolver.cpp` if the new type participates in operations.
- Wire through `CpuDevice` and `AutoSolver` if device dispatch is required.

If the new type is intentionally excluded from some operations (common for bit-special cases), those exclusions must be explicit and tested.

### 5) Python bindings (`src/bindings/` + `src/bindings.cpp`)

- Bind the new class in the correct translation unit (`bind_matrix.cpp` or `bind_vector.cpp`, or `bind_complex.cpp` for complex helpers).
- Ensure the aggregator `src/bindings.cpp` calls the binder.
- Expose constructors, accessors, and key properties (`dtype`, `shape`, etc.).

### 6) Testing (`tests/`)

Verify:

- creation + access,
- persistence (save/load/clone),
- dtype behavior (including bit/int/float boundaries),
- at least one large-ish case that exercises the memory-mapped path.

### 7) Documentation

- Add API reference in `documentation/docs/classes/`.
- Update the relevant guide(s) in `documentation/guides/`.

If the new type changes dtype semantics or bit behavior, add/update internals docs in `documentation/internals/`.
