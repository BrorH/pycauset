# Protocol: Adding New Matrix/Vector Operations

**Objective**: Make new ops easy to add without “hunt across the codebase”, and ensure both **matrix and vector** variants are covered.

## The current architecture (where things go)

1. **Frontend (type resolution + allocation + entry point)**
    - Declarations: `include/pycauset/math/LinearAlgebra.hpp`
    - Definitions: `src/math/LinearAlgebra.cpp`
2. **Context (`ComputeContext`)**
    - `include/pycauset/compute/ComputeContext.hpp`, `src/compute/ComputeContext.cpp`
3. **Dispatcher (`AutoSolver`)**
    - `include/pycauset/compute/AutoSolver.hpp`, `src/compute/AutoSolver.cpp`
4. **Interface (`ComputeDevice`)**
    - `include/pycauset/compute/ComputeDevice.hpp`
5. **Implementations (CPU / CUDA)**
    - CPU device: `include/pycauset/compute/cpu/CpuDevice.hpp`, `src/compute/cpu/CpuDevice.cpp`
    - CPU algorithms: `include/pycauset/compute/cpu/CpuSolver.hpp`, `src/compute/cpu/CpuSolver.cpp`
    - CUDA device: `src/accelerators/cuda/CudaDevice.hpp`, `src/accelerators/cuda/CudaDevice.cu`

## The “Support Checklist” (authoritative)

When adding a new operation, you must decide and/or implement support in each of these axes:

1. **Operand rank**
    - Matrix–Matrix
    - Vector–Vector
    - Matrix–Vector and Vector–Matrix (if applicable)

2. **Scalar kind + flags**
    - Fundamental kinds: `bit`, `int`, `float`
    - Flags/permutations: `complex`, `unsigned`
    - Special rule: **bit is allowed to have op-specific exceptions** (bitwise vs numeric vs error-by-design).
      See [[internals/DType System|internals/DType System]].

3. **Structure/storage**
    - Dense vs triangular vs symmetric/antisymmetric vs identity/diagonal
    - Vector special cases like `UnitVector`

4. **Device coverage**
    - CPU must be correct.
    - GPU is optional; if unsupported, routing must be prevented or it must throw clearly.

5. **Python surface**
    - Bindings: `src/bindings/` (`bind_matrix.cpp`, `bind_vector.cpp`, `bind_complex.cpp`) and the aggregator `src/bindings.cpp`.
    - Python API helpers/wrappers may also need updates in `python/pycauset/`.

6. **Documentation + tests**
    - Add/modify API docs in `documentation/docs/` and guides in `documentation/guides/`.
    - Add tests (Python and/or C++) covering dtype permutations and “error-by-design” cases.

7. **Properties (R1_PROPERTIES)**
    - Decide which properties the op **consumes** (e.g., `is_upper_triangular`, `is_hermitian`).
    - Decide whether the op **produces/propagates/changes** properties on its outputs.
    - Enforce the **no-scan rule**: properties must never be validated by scanning payload data.
    - Document “power user” semantics where relevant: properties are gospel and may override payload truth.

8. **Cached-derived properties (R1_PROPERTIES)**
    - Decide which cached-derived properties (e.g., `trace`) the op **invalidates** (default: invalidate all cached-derived values on any payload mutation).
    - If the op is a metadata-only transform, decide which cached-derived values can be **propagated** via explicit $O(1)$ rules (otherwise clear).
    - If the op is parallelized, decide whether it should emit a constant-size **effect summary** to help the post-op health check update metadata without a second payload pass.

## Step-by-step (what to edit, in order)

### 1) Add the device-interface method

Add a pure virtual method to `include/pycauset/compute/ComputeDevice.hpp`.

Matrix example:
```cpp
virtual void my_new_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
```

Vector example:
```cpp
virtual void my_new_op_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
```

### 2) Implement CPU algorithm + wire CpuDevice

- Add implementation to `include/pycauset/compute/cpu/CpuSolver.hpp` and `src/compute/cpu/CpuSolver.cpp`.
- Wire through `include/pycauset/compute/cpu/CpuDevice.hpp` and `src/compute/cpu/CpuDevice.cpp` as a thin passthrough.

### 3) Add AutoSolver routing

Update `include/pycauset/compute/AutoSolver.hpp` and `src/compute/AutoSolver.cpp`.

- Default: route by size with `select_device(elements)->my_new_op(...)`.
- If GPU does not support the op/dtypes/structures, either:
  - enforce “CPU only” in AutoSolver for that op, or
  - keep GPU routing but ensure GPU throws a clear error.

### 4) (Optional) Implement CUDA

If supported, implement in `src/accelerators/cuda/CudaDevice.cu`.

### 5) Add the frontend wrapper(s)

Add a top-level frontend function in `include/pycauset/math/LinearAlgebra.hpp` + `src/math/LinearAlgebra.cpp`.

Responsibilities of the frontend:
- validate shapes,
- determine the result dtype/structure,
- allocate the result using `ObjectFactory::create_matrix/create_vector`,
- dispatch via `ComputeContext::instance().get_device()->...`.

Additional responsibility (R1_PROPERTIES):

- Compute any **effective structure category** once (e.g., zero/identity/diagonal/triangular/general) from properties and pass that decision down to avoid repeated property lookups in inner loops.

### 6) Bind to Python

Add the binding to the appropriate `src/bindings/bind_*.cpp` and ensure it is reachable from `src/bindings.cpp`.

### 7) Declare dtype/coverage expectations (policy + tests)

Every new op must explicitly state its dtype behavior. In particular:

- **Fundamental-kind rule (bit/int/float):** do not “promote down” across kinds. For example, `matmul(bit, float64) -> float64`.
- **Underpromotion within floats:** `matmul(float32, float64) -> float32` by default.
- **Overflow:** integer overflow throws; large integer matmul may emit a risk warning (advisory).

These rules are defined in [[internals/DType System|internals/DType System]] and summarized in [[project/Philosophy|project/Philosophy]].

## Minimal “Definition of Done” for a new op

- [ ] `ComputeDevice` interface updated.
- [ ] CPU implementation exists and is correct.
- [ ] `AutoSolver` dispatch correct (CPU-only or GPU-enabled).
- [ ] Frontend wrapper(s) in `src/math/LinearAlgebra.cpp` (matrix and/or vector).
- [ ] Python bindings updated.
- [ ] Dtype behavior documented (including bit exceptions and cross-kind rules).
- [ ] Tests cover supported dtypes + at least one “error-by-design” dtype.

If the op is property-aware (R1_PROPERTIES):

- [ ] The op’s behavior with properties is documented (which properties it reads, and which it writes/propagates).
- [ ] Property behavior is deterministic and respects the “no truth validation / no data scans” rule.

If the op interacts with cached-derived properties (R1_PROPERTIES):

- [ ] Cache behavior is documented (which cached-derived values can be preserved/propagated vs must be cleared).
- [ ] Cache behavior is deterministic and uses only $O(1)$ rules (no scans / no extra passes).

For `bit` specifically, always state whether the new op is:

- **bitwise** (stays `bit`, stays packed), or
- **numeric** (may widen to `int`/`float`, potentially huge), or
- **error-by-design** for `bit` unless the user explicitly requests widening.

## See also

- [[project/protocols/Documentation Protocol.md|Documentation Protocol]]
- [[internals/plans/completed/R1_PROPERTIES_PLAN.md|R1_PROPERTIES plan (properties + caches)]]
- [[internals/DType System|internals/DType System]]
