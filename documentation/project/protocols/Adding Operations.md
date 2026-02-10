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

9. **Traits/Tags (Phase 3: property mirroring)**
    - If the op depends on properties for routing, consult the C++ property flags (`include/pycauset/core/PropertyFlags.hpp`).
    - Use `MatrixPropertyTraits` in `AutoSolver` or routing helpers to make decisions without scanning payloads.
    - When adding new boolean properties, update both the Python property map and the C++ flag map.

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

### 8) Register the Op Contract

Every numerical operation must declare its capabilities in the **Op Contract Registry**. This ensures that the system knows which operations support streaming, block matrices, or have special requirements (like square inputs).

**The Contract Struct** (defined in `include/pycauset/core/OpRegistry.hpp`):

```cpp
struct OpContract {
    std::string name;
    bool supports_streaming;     // Can delegate to StreamingManager?
    bool supports_streaming;     // Does it delegate to StreamingManager?
    bool supports_block_matrix;  // Does it natively handle BlockMatrix recursion?
    bool requires_square;        // Does it enforce square inputs?
    // Future: SIMD Tier, Property propagation rules
};
```

### Usage (C++)

```cpp
// On module load / static init
OpContract matmul_contract;
matmul_contract.name = "matmul";
matmul_contract.supports_streaming = true;
matmul_contract.supports_block_matrix = true;

OpRegistry::instance().register_op(matmul_contract);
```

### Usage (Python)

The registry is mirrored to Python for inspection and AI-guided development.

```python
import _pycauset as internal  # or pycauset.internals depending on binding layout

contract = internal.OpRegistry.instance().get_contract("matmul")
print(contract.supports_streaming) # True
```

### Integrating a New Op

1. Define the C++ implementation.
2. In the initialization static block, register the `OpContract`.

## GPU routine authoring checklist (Phase 2, R1_GPU)

Use this checklist when you add a **GPU acceleration routine** (host-orchestrated driver) for a new op. This is the plug-and-play contract that keeps GPU work predictable and future R1_CPU-compatible.

1. **Define the Driver contract** (host-side orchestration).
    - Inputs: operand shapes, dtype, memory layout, and streaming tile constraints.
    - Outputs: explicit success/failure codes and error messages (no silent fallbacks).
    - Dependencies: the driver may call `AsyncStreamer` and device kernels, but **must not** embed device-specific logic in the orchestration loop.

2. **Declare capability surface (routing gate).**
    - Enumerate supported dtypes and matrix structures.
    - State any property preconditions (e.g., symmetric/triangular).
    - Provide a “no” reason string for unsupported cases (for routing traces).

3. **Use the MemoryGovernor budget.**
    - All pinned allocations must request a ticket from the governor.
    - If denied, degrade to pageable host buffers (or CPU fallback) and record a trace reason.

4. **Provide CPU compatibility.**
    - The driver contract must remain backend-agnostic so R1_CPU can reuse the same orchestration loop.
    - Device-specific kernels should sit behind a worker interface; do not bake CUDA into the driver.

5. **Instrument routing and execution.**
    - Emit trace tags for routing choice, streaming decisions, and kernel/driver stages.
    - Ensure observability is deterministic and does not depend on timing.

6. **Register the driver in AutoSolver.**
    - Add the routing decision (cost model + capability checks) in `AutoSolver`.
    - Fall back to CPU with a clear reason if the driver cannot run.

7. **Document & test.**
    - Update the API reference and guides with the new acceleration path.
    - Add correctness tests and at least one routing test (GPU chosen / GPU rejected).

## GPU kernel checklist (Phase 5, R1_GPU)

Use this when adding or modifying **low-level CUDA kernels** (beyond the host driver):

1. **Declare kernel coverage.**
    - Supported dtypes, structures, and any layout assumptions (row-major/col-major, contiguous, stride rules).
    - Explicitly document unsupported permutations and how routing rejects them.

2. **Validate inputs.**
    - Bounds/shape checks must happen before launching kernels.
    - Fast-fail with a clear error on invalid sizes or unsupported structure flags.

3. **Precision and determinism.**
    - Declare math mode (e.g., TF32 allowed vs disabled).
    - Document any non-deterministic reductions or race-sensitive paths.

4. **Memory and stream discipline.**
    - Honor the MemoryGovernor pinning budget for host staging.
    - Use the caller’s stream (no implicit default-stream races).
    - Avoid hidden device allocations in hot loops.

5. **Error handling and traceability.**
    - Check CUDA errors (`cudaGetLastError`, `cublasStatus`, `cusolverStatus`).
    - Emit deterministic trace tags for routing and kernel stages.

6. **Tests.**
    - Add correctness tests for representative shapes/dtypes.
    - Add a routing test that confirms GPU selection for supported inputs.
    - Add a failure test for a known unsupported permutation.

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
- [[internals/Compute Architecture.md|Compute Architecture]]
- [[internals/Streaming Manager.md|Streaming Manager]]
