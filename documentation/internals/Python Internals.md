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

## `_internal/` modules

| Module | Owns |
|---|---|
| `native.py` | importing the extension (Windows DLL paths); the thread budget is the engine's (`ParallelUtils`, `init_thread_budget`) |
| `runtime.py` | backing-directory roots and temporary-file cleanup at exit |
| `ops.py` | the glue of the linear algebra entry points (`solve`, `lstsq`, `pinv`, `norm`, ...), each traced for `last_io_trace` |
| `matrix_api.py`, `factories.py`, `coercion.py`, `dtypes.py` | constructing matrices and vectors from data, dtype tokens |
| `formatting.py` | `repr` / `str` added to the native classes |
| `export_guard.py`, `native_threshold.py` | export to NumPy and its size guard; scoped memory-threshold changes |
| `persistence.py` | the `.pycauset` container: save, load, metadata |
| `io_observability.py` | IO traces (`last_io_trace`) |
| `blockmatrix.py`, `submatrix_view.py`, `thunks.py`, `dense_python.py` | block matrices: blocks, windows into them, lazy block products |
| `warnings.py` | the warning categories |

If you’re unsure where a new helper belongs, prefer `_internal/` first.

## How to add a new top-level function safely

1. Implement the logic in an `_internal/` module.
2. Add a thin wrapper/re-export at `pycauset.<name>` in `python/pycauset/__init__.py`.
3. Add or update a test under `tests/engine/` (usually `test_api_surface.py`) that uses the symbol through `pycauset.*`.
4. Add its doc page under `documentation/docs/` (`test_api_surface.py` fails for a public name without one).
5. If the feature needs a new native binding, follow [[internals/Bindings & Dispatch]].

## Status

`__init__.py` is still large (about 1,500 lines); splitting it is review PLAN 4.4, after the
Python patch stack goes (4.3).
