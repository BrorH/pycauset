# Project Protocols

This document serves as the authoritative guide for contributing to PyCauset. It covers the core philosophy, development workflows, and release procedures.

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

**Objective**: Ensure all new mathematical operations follow the unified `ComputeContext` architecture, maintaining hardware abstraction and automatic dispatch.

### The Hierarchy

1.  **Frontend (`MatrixOperations.cpp`)**: Type resolution, result allocation, and entry point.
2.  **Context (`ComputeContext`)**: Singleton managing the active device.
3.  **Dispatcher (`AutoSolver`)**: Decides CPU vs. GPU based on size/availability.
4.  **Interface (`ComputeDevice`)**: Abstract base class defining the operation contract.
5.  **Implementations (`CpuDevice`/`CudaDevice`)**: Hardware-specific logic.

### Step-by-Step Implementation Guide

#### 1. Define the Interface (`include/ComputeDevice.hpp`)
Add a pure virtual method for your operation.
```cpp
virtual void my_new_op(const MatrixBase& a, const MatrixBase& b, MatrixBase& result) = 0;
```

#### 2. Implement CPU Logic (`src/CpuSolver.cpp` & `include/CpuSolver.hpp`)
This is the "reference" implementation and fallback.
*   **Header**: Add the method declaration.
*   **Source**: Implement the algorithm.
    *   Use `ParallelFor` from `ParallelUtils.hpp` for threading.
    *   Handle different matrix types (Dense, Triangular, etc.) using `dynamic_cast` or templates.
    *   **Crucial**: Ensure it handles `float`, `double`, and `int` (or throws if unsupported).

#### 3. Wire up the CPU Device (`src/CpuDevice.cpp` & `include/CpuDevice.hpp`)
`CpuDevice` is just a wrapper. Pass the call through to `CpuSolver`.
```cpp
void CpuDevice::my_new_op(...) {
    solver_.my_new_op(a, b, result);
}
```

#### 4. Update the Dispatcher (`src/AutoSolver.cpp` & `include/AutoSolver.hpp`)
Implement the routing logic.
*   **Default**: Delegate to `select_device(elements)->my_new_op(...)`.
*   **Custom**: If the operation is *never* supported on GPU, just call `cpu_device_->my_new_op(...)`.

#### 5. (Optional) Implement GPU Logic (`src/accelerators/cuda/CudaDevice.cpp`)
If the operation can be accelerated:
*   Add the method to `CudaDevice`.
*   Implement using `cuBLAS`, `cuSOLVER`, or custom kernels.
*   **Important**: If you don't implement it in `CudaDevice`, you *must* ensure `AutoSolver` never routes this operation to the GPU, OR implement a fallback in `CudaDevice` that throws or copies back to CPU (not recommended for performance).

#### 6. Create the Frontend (`src/MatrixOperations.cpp`)
This is what the Python bindings call.
1.  **Resolve Types**: Use `MatrixFactory::resolve_result_type`.
2.  **Create Result**: `MatrixFactory::create(...)`.
3.  **Dispatch**:
    ```cpp
    ComputeContext::instance().get_device()->my_new_op(a, b, *result);
    ```

#### 7. Bind to Python (`src/bindings.cpp`)
Expose the frontend function to Python via `pybind11`.

### Checklist

- [ ] Added to `ComputeDevice` interface?
- [ ] Implemented in `CpuSolver` with parallelization?
- [ ] Wired through `CpuDevice`?
- [ ] Added dispatch logic to `AutoSolver`?
- [ ] Created frontend wrapper in `MatrixOperations`?
- [ ] Added unit tests in `tests/`?

---

## 3. Protocol: Adding New Matrix Types

**Objective**: Ensure new matrix structures (e.g., Symmetric, Sparse) are fully integrated into the factory, storage, compute, and Python ecosystems.

### Step-by-Step Implementation Guide

#### 1. Core Definitions (`include/pycauset/core/Types.hpp`)
*   Add a new enum value to `MatrixType`.

#### 2. Define the Class (`include/pycauset/matrix/`)
*   Create a new header file (e.g., `MyMatrix.hpp`).
*   Inherit from `MatrixBase` (or a specialized base like `TriangularMatrixBase`).
*   Implement required virtual methods:
    *   `get_element_as_double(i, j)`
    *   `multiply_scalar`, `add_scalar`, `transpose`
    *   `clone` (usually delegates to `ObjectFactory`)
*   **Expose Raw Memory**: If the matrix is intended for GPU acceleration or BLAS operations, it MUST expose a `data()` method returning a raw pointer (`T*`) to the underlying memory. This is critical for `cudaMemcpy` and interoperability.
*   Implement storage access logic (`get`, `set`) using `MemoryMapper`.
    *   **Crucial**: Ensure pointer arithmetic correctly handles byte vs. bit offsets.

#### 3. Update the Factory (`src/core/ObjectFactory.cpp`)
The factory is the central registry for creating and loading matrices.
*   **`create_matrix`**: Add a case for your `MatrixType` enum. Initialize the object with `create_new=true`.
*   **`load_matrix`**: Add a case to reconstruct the object from an existing file (`create_new=false`).
*   **`clone_matrix`**: Ensure the new type can be deep-copied.

#### 4. Update the Solver (`src/compute/cpu/CpuSolver.cpp`)
The CPU solver handles mathematical operations.
*   **`binary_op_impl`**: Add a `dynamic_cast` block to handle your new type as a *result* of operations (e.g., `A + B -> NewType`).
*   **Specific Operations**: If your matrix requires specialized algorithms (e.g., Cholesky for Symmetric), implement them in `CpuSolver`.

#### 5. Python Bindings (`src/bindings.cpp`)
*   Bind the class using `pybind11`.
*   Expose constructors, accessors, and properties.
*   Update `cast_matrix_result` to allow returning this type to Python.
*   Bind arithmetic operations (if applicable).

#### 6. Testing (`tests/`)
*   Create a dedicated test file (e.g., `test_my_matrix.cpp`).
*   Verify:
    *   Creation and Access (`get`/`set`).
    *   Persistence (Save/Load).
    *   Arithmetic (Add/Multiply) - check for `inf`/`nan` which often indicate offset errors.

#### 7. Documentation
*   Add API reference in `documentation/docs/classes/`.
*   Update `documentation/docs/classes/index.md`.

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
