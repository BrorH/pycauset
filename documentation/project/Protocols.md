# Project Protocols

This document serves as the authoritative guide for contributing to PyCauset. It covers the core philosophy, development workflows, and release procedures.

## 0. Pre-alpha Policy (Scope + Approval Gates)

PyCauset is currently **pre-alpha** and effectively single-maintainer. There is no external userbase to preserve yet.

- Backward compatibility is **not** a hard constraint right now.
- Breaking changes to the Python surface and/or architecture are allowed **when they improve the overall approach**.
- Approval gate: before changing the public Python surface (names/semantics) or making a large architectural shift, propose the change + tradeoffs and wait for explicit approval.
- If a breaking change is approved, update tests and documentation in the same change set (don’t leave the repo in a “half-migrated” state).

## 1. Documentation Protocol

**Objective:** Create a comprehensive, book-like resource for users. Documentation is not just a reference; it is a teaching tool.

### Core Philosophy
*   **Redundancy is Required:** Do not document in only one place. Central concepts (e.g., Eigenvectors) belong in the API reference, the User Manual, and relevant Guides.
*   **Teach, Don't Just List:** Write to instruct. Explain *how* to use a feature and *why*, not just what it is.
*   **Write More Than You Think:** Err on the side of over-explanation.
*   **Modify, Don't Just Append:** If a feature fundamentally changes a concept, rewrite the existing guide sections rather than tacking on a paragraph at the end.

### Documentation Locations

#### A. API Reference (`documentation/docs/`)
*Mandatory for ALL code additions (Classes, Functions, Parameters).*

*   **Classes (`docs/classes/`)**:
    *   If you add/modify a class (e.g., `Matrix`, `Spacetime`), update the corresponding file in this folder.
    *   Document methods, attributes, and usage examples.
*   **Functions (`docs/functions/`)**:
    *   New standalone functions go here.
*   **Parameters (`docs/parameters/`)**:
    *   If a new global parameter or configuration option is added, document it here.

#### B. User Guides (`documentation/guides/`)
*Mandatory for user-facing features.*

*   **Purpose**: The "How-To" manual.
*   **Action**:
    *   **Existing Guides**: If the feature relates to an existing topic (e.g., Spacetime), update `guides/Spacetime.md`. Integrate the new info naturally into the flow.
    *   **New Guides**: If the feature is a new domain, create a new guide.

#### C. Internals (`documentation/internals/`)
*Mandatory for backend/architectural changes.*

*   **Purpose**: For developers and future AI agents.
*   **Content**: File formats, memory architecture, algorithms.

### The "Definition of Done" Checklist

Before marking a task as complete, verify:

1.  [ ] **API Reference Updated?** (Did I add the function/class to `docs/`?)
2.  [ ] **User Guide Updated?** (Did I explain *how* to use this in `guides/`? Did I update existing guides if this changes behavior?)
3.  [ ] **Cross-Referenced?** (Is this feature mentioned in all relevant contexts? e.g., Vector math AND Matrix operations?)
4.  [ ] **Internals Documented?** (If I changed the backend, is it in `internals/`?)
5.  [ ] **Quality Check**: Is it written as a guide for a human, not just a machine-readable list?

---

## 2. Protocol: Adding New Matrix/Vector Operations

**Objective**: Make new ops easy to add without “hunt across the codebase”, and ensure both **matrix and vector** variants are covered.

### The current architecture (where things go)

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

### The “Support Checklist” (this is the authoritative list)

When adding a new operation, you must decide and/or implement support in each of these axes:

1. **Operand rank**
    - Matrix–Matrix
    - Vector–Vector
    - Matrix–Vector and Vector–Matrix (if applicable)

2. **Scalar kind + flags**
    - Fundamental kinds: `bit`, `int`, `float`
    - Flags/permutations: `complex`, `unsigned`
    - Special rule: **bit is allowed to have op-specific exceptions** (bitwise vs numeric vs error-by-design). See `documentation/internals/DType System.md`.

3. **Structure/storage**
    - Dense vs triangular vs symmetric/antisymmetric vs identity/diagonal (and vector special cases like `UnitVector`).

4. **Device coverage**
    - CPU must be correct.
    - GPU is optional; if unsupported, routing must be prevented or it must throw clearly.

5. **Python surface**
    - Bindings: `src/bindings/` (`bind_matrix.cpp`, `bind_vector.cpp`, `bind_complex.cpp`) and the aggregator `src/bindings.cpp`.
    - Python API helpers/wrappers may also need updates in `python/pycauset/`.

6. **Documentation + tests**
    - Add/modify API docs in `documentation/docs/` and guides in `documentation/guides/`.
    - Add tests (Python and/or C++) covering dtype permutations and “error-by-design” cases.

### Step-by-step (what to edit, in order)

#### 1) Add the device-interface method

Add a pure virtual method to `include/pycauset/compute/ComputeDevice.hpp`.

Matrix example:
```cpp
virtual void my_new_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
```

Vector example:
```cpp
virtual void my_new_op_vector(const VectorBase& a, const VectorBase& b, VectorBase& result) = 0;
```

#### 2) Implement CPU algorithm + wire CpuDevice

- Add implementation to `include/pycauset/compute/cpu/CpuSolver.hpp` and `src/compute/cpu/CpuSolver.cpp`.
- Wire through `include/pycauset/compute/cpu/CpuDevice.hpp` and `src/compute/cpu/CpuDevice.cpp` as a thin passthrough.

#### 3) Add AutoSolver routing

Update `include/pycauset/compute/AutoSolver.hpp` and `src/compute/AutoSolver.cpp`.

- Default: route by size with `select_device(elements)->my_new_op(...)`.
- If GPU does not support the op/dtypes/structures, either:
  - enforce “CPU only” in AutoSolver for that op, or
  - keep GPU routing but ensure GPU throws a clear error.

#### 4) (Optional) Implement CUDA

If supported, implement in `src/accelerators/cuda/CudaDevice.cu`.

#### 5) Add the frontend wrapper(s)

Add a top-level frontend function in `include/pycauset/math/LinearAlgebra.hpp` + `src/math/LinearAlgebra.cpp`.

Responsibilities of the frontend:
- validate shapes,
- determine the result dtype/structure,
- allocate the result using `ObjectFactory::create_matrix/create_vector`,
- dispatch via `ComputeContext::instance().get_device()->...`.

#### 6) Bind to Python

Add the binding to the appropriate `src/bindings/bind_*.cpp` and ensure it is reachable from `src/bindings.cpp`.

#### 7) Declare dtype/coverage expectations (policy + tests)

Every new op must explicitly state its dtype behavior. In particular:

- **Fundamental-kind rule (bit/int/float):** do not “promote down” across kinds. For example, `matmul(bit, float64) -> float64`.
- **Underpromotion within floats:** `matmul(float32, float64) -> float32` by default.
- **Overflow:** integer overflow throws; large integer matmul may emit a risk warning (advisory).

These rules are defined in `documentation/internals/DType System.md` and summarized in `documentation/project/Philosophy.md`.

### Minimal “Definition of Done” for a new op

- [ ] `ComputeDevice` interface updated.
- [ ] CPU implementation exists and is correct.
- [ ] `AutoSolver` dispatch correct (CPU-only or GPU-enabled).
- [ ] Frontend wrapper(s) in `src/math/LinearAlgebra.cpp` (matrix and/or vector).
- [ ] Python bindings updated.
- [ ] Dtype behavior documented (including bit exceptions and cross-kind rules).
- [ ] Tests cover supported dtypes + at least one “error-by-design” dtype.

For `bit` specifically, always state whether the new op is:

- **bitwise** (stays `bit`, stays packed), or
- **numeric** (may widen to `int`/`float`, potentially huge), or
- **error-by-design** for `bit` unless the user explicitly requests widening.

---

## 3. Protocol: Adding New Matrix and Vector Types

**Objective**: Ensure new matrix/vector structures are fully integrated into the factory, storage, compute, and Python ecosystems.

### Step-by-Step Implementation Guide

#### 1) Core definitions (`include/pycauset/core/Types.hpp`)

- Add a new enum value to `MatrixType`.
- Decide whether the new type is a matrix-like (`rows x cols`) or a vector-like (`n x 1`) object.

#### 2) Define the class

- Matrix types live in `include/pycauset/matrix/`.
- Vector types live in `include/pycauset/vector/`.

Implement the required virtual interface:

- Matrices (`MatrixBase`): `get_element_as_double(i, j)`, `multiply_scalar`, `add_scalar`, `transpose`, and correct `clone()` behavior.
- Vectors (`VectorBase`): `get_element_as_double(i)`, `multiply_scalar`, `add_scalar`, `transpose` (if supported), and correct `clone()` behavior.

If the type is intended for GPU/BLAS interoperability, it must expose contiguous raw memory access where appropriate.

#### 3) Update the factory (`include/pycauset/core/ObjectFactory.hpp`, `src/core/ObjectFactory.cpp`)

The factory is the central registry for creating, loading, and cloning persistent objects.

- Matrices:
    - `create_matrix`
    - `load_matrix`
    - `clone_matrix`
- Vectors:
    - `create_vector`
    - `load_vector`
    - `clone_vector`

#### 4) Update compute support

- Add CPU support in `src/compute/cpu/CpuSolver.cpp` if the new type participates in operations.
- Wire through `CpuDevice` and `AutoSolver` if device dispatch is required.

If the new type is intentionally excluded from some operations (common for bit-special cases), those exclusions must be explicit and tested.

#### 5) Python bindings (`src/bindings/` + `src/bindings.cpp`)

- Bind the new class in the correct translation unit (`bind_matrix.cpp` or `bind_vector.cpp`, or `bind_complex.cpp` for complex helpers).
- Ensure the aggregator `src/bindings.cpp` calls the binder.
- Expose constructors, accessors, and key properties (`dtype`, `shape`, etc.).

#### 6) Testing (`tests/`)

Verify:

- creation + access,
- persistence (save/load/clone),
- dtype behavior (including bit/int/float boundaries),
- at least one large-ish case that exercises the memory-mapped path.

#### 7) Documentation

- Add API reference in `documentation/docs/classes/`.
- Update the relevant guide(s) in `documentation/guides/`.

If the new type changes dtype semantics or bit behavior, add/update internals docs in `documentation/internals/`.

---

## 4. Release Process

PyCauset uses an automated release pipeline based on GitHub Actions. This document outlines how versioning and publishing to PyPI are handled.

### Automated Releases

The primary way to release a new version is by pushing to the `main` branch.

1.  **Commit your changes**: Ensure your work is committed.
2.  **Push to main**:
    ```bash
    git push origin main
    ```
3.  **Workflow Trigger**: The GitHub Action `Publish to PyPI` will start automatically.
    *   **Bump Version**: It calculates the next **patch** version (e.g., `0.2.4` -> `0.2.5`).
    *   **Tag**: It creates a new git tag and a GitHub Release using your commit message as the notes.
    *   **Build & Publish**: It builds wheels for Windows, macOS, and Linux, and publishes them to PyPI.

### Manual Releases

If you need to bump a minor/major version or set a specific version number, you can use the helper script or git tags directly.

#### Using the Helper Script (`release.ps1`)

A PowerShell script is provided in the root directory to simplify manual tagging.

**Bump Minor Version:**
```powershell
.\release.ps1 -Type minor
```
*Example: `0.2.4` -> `0.3.0`*

**Bump Major Version:**
```powershell
.\release.ps1 -Type major
```
*Example: `0.2.4` -> `1.0.0`*

**Set Specific Version:**
```powershell
.\release.ps1 -SetVersion 0.4.0
```
*Example: Jumps directly to `0.4.0`*

#### Using Git Tags

You can also manually create and push a tag. The CI/CD pipeline will detect the new tag and build/publish that specific version (skipping the auto-bump step).

```bash
git tag v0.4.0
git push origin v0.4.0
```

### CI/CD Workflow Details

The workflow is defined in `.github/workflows/publish.yml`.

*   **Triggers**:
    *   `push` to `main`: Triggers auto-bump (patch) and release.
    *   `push` of tags (`v*`): Triggers build and release for that tag.
    *   `workflow_dispatch`: Allows manual triggering from GitHub Actions UI.
*   **Versioning**: Uses `setuptools_scm` to determine the package version from git tags.
*   **Build System**: Uses `cibuildwheel` to build binary wheels for multiple platforms.
*   **Publishing**: Uses PyPI Trusted Publishing (OIDC) to upload artifacts.

---

## 5. Testing and Bug Tracking Protocol

**Objective:** Maintain a strict protocol for automated testing and bug tracking to ensure code quality and stability.

### Bug Documentation
Whenever a bug is discovered—whether it's a compilation error, a runtime failure, a logic error, or a regression—it **MUST** be documented in `tests/BUG_LOG.md`.

#### Format
Append a new entry to `tests/BUG_LOG.md` using the following format:

```markdown
## [Date: YYYY-MM-DD HH:MM] Bug Title

**Status**: [Fixed / Open]
**Severity**: [Critical / High / Medium / Low]
**Component**: [e.g., Storage, Matrix Operations, Python Bindings]

**Description**:
A concise description of the issue.

**Reproduction**:
Steps or code snippet to reproduce the failure.

**Root Cause** (if known):
Technical explanation of why it happened.

**Fix** (if applied):
Description of the solution implemented.
```

### Test Creation vs. Execution
When working on tests, follow this workflow:
1.  **Design**: Create comprehensive test cases covering edge cases, boundary conditions, and type permutations.
2.  **Review**: Ensure tests are valid before running.
3.  **Execute**: Run tests and monitor for failures.

### Regression Prevention
When fixing a bug, ensure a regression test is added to the test suite (preferably in a dedicated `test_regressions.py` or the relevant module) to prevent the issue from recurring.
