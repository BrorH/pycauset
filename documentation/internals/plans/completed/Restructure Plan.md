# Codebase Restructure Plan (Proposal + Execution Record)

**Status:** Partially executed. Remaining work requires explicit approval.

## Execution status (as of 2025-12-14)

This document started as a proposal. Since then, several phases have been executed in this repo.

For contributors: the bullets below are the “what happened” summary. The canonical details live in the linked dev handbook pages.

- [x] Phase A — Documentation-first
  - `documentation/dev/` handbook exists (bootstrap/build/bindings/testing/hygiene/structure).
  - Philosophy positioning updated to “NumPy for causal sets”.
  - Documented in: [[dev/index]], [[project/Philosophy]]
- [x] Phase B — Purge committed binaries + enforce hygiene
  - Committed build artifacts/binaries were removed and history rewritten.
  - Documented in: [[dev/Repository Hygiene]]
- [x] Phase C — Build workflow alignment
  - Pip/scikit-build-core workflow is the documented “canonical” path; scripts are wrappers.
  - Documented in: [[dev/Build System]]
- [~] Phase D — Python internal modularization
  - Public API remains `pycauset.*`.
  - `python/pycauset/_internal/` created and used for implementation.
  - Persistence (single-file container) + linalg caching extracted.
  - Runtime/storage/temp-file policy + patching + factories + formatting extracted.
  - Ops glue (`matmul`, `compute_k`, `bitwise_not`, `invert`) extracted into `python/pycauset/_internal/ops.py` and `__init__.py` delegates.
  - Remaining work: keep shrinking `python/pycauset/__init__.py` and keep dev docs in sync.
  - Documented in: [[dev/Codebase Structure]], [[dev/Python Internals]]
- [~] Phase E — Bindings modularization
  - `src/bindings.cpp` is a thin `PYBIND11_MODULE` entrypoint.
  - Binding code split into modular translation units under `src/bindings/`.
  - Added native export drift check: `tools/check_native_exports.py`.
  - Documented in: [[dev/Bindings & Dispatch]]
- [ ] Phase F — NxM groundwork
  - Square-only assumptions list started: `documentation/dev/Square-only Assumptions.md`.
  - Documented in: [[dev/Square-only Assumptions]]

## 0) Executive summary

PyCauset is a large hybrid project (Python API + C++ core + optional CUDA). The top priority is to make the codebase **predictable to navigate and safe to modify**, while preserving the core philosophy:

- **PyCauset is “NumPy for causal sets”.** Users interact with top-level Python objects and functions (e.g., `pycauset.matrix`, `pycauset.causal_matrix`, `pycauset.matmul`).
- **Optimizations happen behind the scenes**: tiered storage, device dispatch, direct-vs-streaming, SIMD/BLAS, etc.
- We may reorganize internal folders/modules, but we **must not** push user entrypoints into subpackages like `pycauset.physics.*`.

This plan focuses on:
1) making builds reproducible and eliminating “stale binary” confusion,
2) clarifying ownership boundaries between subsystems,
3) writing enough documentation that new contributors can make changes safely,
4) keeping future requirements in mind (notably: **NxM matrices** for all types; **no N-D arrays**).

---

## 1) Goals

- **Maintainability:** a contributor can answer “where does this belong?” quickly.
- **API coherence:** the top-level Python surface keeps a NumPy-like feel (entrypoints remain at `pycauset.*`).
- **Pre-alpha flexibility:** breaking changes are acceptable when they improve the architecture, but they require explicit approval and corresponding updates to tests + docs.
- **Reproducibility:** the code you run is the code you built.
- **Documentation completeness:** “no amount is too much” — developers should have transparent guides.
- **Extensibility:** adding dtype/op support can follow a clear recipe (ties into optimization checklist).

## 2) Non-goals (for this restructure)

- Implementing new algorithms or performance changes (except as necessary to keep things building).
- Introducing N-dimensional array semantics (explicitly out of scope).
- Large user-visible API redesign.

## 3) Constraints / invariants

### Public API invariants
- End-user entrypoints remain **top-level**: `pycauset.*` (do not push the primary surface behind submodules like `pycauset.physics.*`).
- Internal reorg is allowed if `pycauset.__init__` re-exports the intended public symbols.
- Pre-alpha policy: the public surface may change, but only with explicit approval and synchronized updates to tests + docs.

### Roadmap invariants
- Future direction: support **NxM matrices for all types** (dense, triangular, symmetric, bit, etc.).
- Still no arbitrary N-D arrays.

### Repo hygiene invariants
- Do **not** commit compiled artifacts (e.g., `_pycauset.pyd`, `.dll`, `.so`) into the repo.
- The canonical build path is pip/scikit-build-core (per `pyproject.toml`).

---

## 4) Phased execution plan

Each phase has an explicit “Done when…” acceptance criterion.

### Phase A — Documentation-first (developer transparency)

**Work:**
- Create a dedicated `documentation/dev/` handbook (this folder).
- Add missing developer docs:
  - overall codebase structure,
  - build system explanation,
  - bindings/dispatch guide,
  - testing/benchmarks guide,
  - repository hygiene rules.
- Update `documentation/project/Philosophy.md` so the first-order philosophy is explicitly “NumPy for causal sets”.

**Done when:**
- A new contributor can follow docs to:
  - find the implementation of a top-level API call,
  - add a new dtype/op in the correct places,
  - run tests/benchmarks.

### Phase B — Purge committed binaries + enforce hygiene

**Work:**
- Remove currently committed compiled artifacts from the repo (e.g., binaries under the Python package directory).
- Add ignore rules so these never get reintroduced.
- Rewrite git history to purge these artifacts (safe because it is a single-maintainer repo).

**Done when:**
- Fresh clone contains only source + docs (no compiled artifacts).
- Building/installing produces artifacts locally.

### Phase C — Build workflow alignment (pip as source of truth)

**Work:**
- Keep `build.ps1` as a thin wrapper that calls the canonical pip build/install commands.
- Ensure compiler flags/warning suppressions remain in CMake and are not lost (pip uses CMake via scikit-build-core).
- Document how to pass common build options:
  - Release vs Debug,
  - enabling CUDA,
  - setting CMake cache args.

**Done when:**
- “The official way” to build from source is documented as pip-based.
- `build.ps1` cannot diverge into a second, incompatible build system.

### Phase D — Python package internal modularization (without changing public API)

**Work (internal-only):**
- Split the current large `pycauset` package internals by responsibility (example target shape):
  - `_runtime/` (platform/bootstrap: DLL search paths, environment checks)
  - `_native/` (native import helpers and thin wrappers)
  - `storage/` (file formats, save/load, storage roots)
  - `linalg/` (Python-facing linear algebra helpers and high-level glue)
  - `physics/` (CausalSet, sprinkling, spacetimes, fields)
  - `vis/` (plotly integrations)
- Keep user entrypoints **top-level** via re-export in `pycauset/__init__.py`.

**Done when:**
- `pycauset/__init__.py` is a small, readable facade.
- There is exactly one canonical place for:
  - persistence logic,
  - compute config,
  - native importing/bootstrap.
- Tests continue to import from `pycauset.*` unchanged.

### Phase E — Bindings completeness + modular binding sources

**Work:**
- Make binding code modular (multiple binding translation units) to match subsystems.
- Ensure Python expectations and native exports do not drift.
- Add a “binding coverage checklist” doc (what symbols are required, where they come from).

**Done when:**
- A mismatch between Python expectations and C++ bindings is easy to detect.
- Adding a new matrix type/op has a clear binding template.

### Phase F — NxM groundwork (documentation + interfaces first)

**Work:**
- Update roadmap/TODO: NxM support planned for all types.
- Identify (document-only initially) which components assume square matrices today:
  - storage metadata,
  - matrix base classes,
  - solvers (matmul/inverse),
  - Python factories.

**Done when:**
- The codebase has a documented “square-only assumptions list”.
- Future NxM work can proceed systematically.

---

## 5) Risk management

- **History rewrite risk:** force-push breaks old clones. Mitigation: since single-maintainer, do it once, then re-clone locally.
- **API drift risk:** internal reorg can accidentally change user imports. Mitigation: keep stable top-level re-exports and run interface tests.
- **Build drift risk:** multiple build scripts can diverge. Mitigation: make scripts wrappers + document canonical commands.

---

## 6) Deliverables checklist

- [ ] New `documentation/dev/` handbook exists and is referenced by contributors.
- [ ] Philosophy updated to lead with “NumPy for causal sets”.
- [ ] Repo purge of binaries + `.gitignore` enforcement.
- [ ] Build scripts are wrappers; CMake flags preserved.
- [ ] Python internals modularized with stable `pycauset.*` surface.
- [ ] Bindings modular + documented.
- [ ] NxM roadmap noted and square-only assumptions documented.

---

## 7) Approval gates (explicit stop points)

- **Gate 1:** Approve documentation structure + file list.
- **Gate 2:** Approve history rewrite (purge binaries).
- **Gate 3:** Approve internal Python package reorg target structure.
- **Gate 4:** Approve bindings reorg.

Until each gate is approved, the work must remain read-only or documentation-only.
