# Python Internals (How the public API stays clean)

PyCauset’s public Python surface is intentionally small and NumPy-like: users import and call `pycauset.*`.

To keep that surface stable while still allowing rapid refactors, most implementation code lives under:

- `python/pycauset/_internal/`

Nothing in `_internal/` should be treated as a public API.

## Design rules

- `python/pycauset/__init__.py` is the **public facade**.
- `_internal/` contains **implementation modules** that the facade delegates to.
- When adding functionality, default to putting helper logic into `_internal/` and re-exporting only the intended entrypoint at `pycauset.*`.
- If you change the public surface (names, semantics, import paths), treat it as a major decision: get explicit approval and update tests + docs.

## Current `_internal/` modules (what they own)

These modules exist today and are the canonical homes for their responsibilities:

- `python/pycauset/_internal/runtime.py`
  - Runtime/bootstrap policy (platform checks, environment setup).
- `python/pycauset/_internal/native.py`
  - Native import helpers and thin wrappers around `pycauset._pycauset`.
- `python/pycauset/_internal/persistence.py`
  - Persistence and storage helpers used by the public API.
- `python/pycauset/_internal/linalg_cache.py`
  - Linear algebra caching/glue (Python-level).
- `python/pycauset/_internal/factories.py`
  - Object construction helpers (Matrix/Vector creation, convenience factories).
- `python/pycauset/_internal/ops.py`
  - Operation “glue” that keeps the top-level API thin.
  - Examples (as of today): `matmul`, `compute_k`, `bitwise_not`, `invert`.
- `python/pycauset/_internal/coercion.py`
  - Argument coercion and dtype-like normalization for Python entrypoints.
- `python/pycauset/_internal/patching.py`
  - Patch/update helpers (used by persistence and runtime/storage workflows).
- `python/pycauset/_internal/formatting.py`
  - String formatting / repr helpers.
- `python/pycauset/_internal/matrix_api.py`
  - Python-side Matrix API helpers (methods/properties that wrap native behavior).

If you’re unsure where a new helper belongs, prefer `_internal/` first.

## How to add a new top-level function safely

1. Implement the logic in an `_internal/` module.
2. Add a thin wrapper/re-export at `pycauset.<name>` in `python/pycauset/__init__.py`.
3. Add or update a Python test under `tests/python/` that imports the symbol from `pycauset.*`.
4. If the feature relies on native bindings, update [[dev/Bindings & Dispatch]] and run the drift check (`python tools/check_native_exports.py`).

## Status note

The repo has already moved a lot of logic into `_internal/`. Ongoing work should continue shrinking `python/pycauset/__init__.py` into a readable facade and keeping this doc aligned with the current layout.
