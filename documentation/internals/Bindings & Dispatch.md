# Bindings & Dispatch

How Python calls reach the C++ engine. The engine-side path (entry function, device,
kernel) is [[internals/Compute Architecture]].

## Layers

- **`pycauset.*`** (`python/pycauset/__init__.py`): the public names. Thin: argument
  checks, then the engine.
- **`pycauset._pycauset`**: the pybind11 module. `src/bindings.cpp` is the
  `PYBIND11_MODULE` entry; the bindings live in `src/bindings/`, one file per subsystem:

| File | Binds |
|---|---|
| `bind_core.cpp` | threads, memory threshold, `sum`, the lazy switch, kernel / IO / access traces |
| `bind_matrix.cpp` | the matrix classes, operators, NumPy export |
| `matrix_indexing.hpp` | matrix indexing: subscript parsing, slicing, element get / set through views, `__getitem__` / `__setitem__` |
| `bind_vector.cpp` | the vector classes |
| `bind_math.cpp` | linear algebra and reductions |
| `bind_causet.cpp` | the Minkowski diamond and the sprinkler |
| `numpy_import.hpp` | NumPy arrays as operands (one importer for every binding) |
| `numpy_export_guard.hpp` | the size guard on export to NumPy |

## Rules

- **One binding helper per operation.** Every spelling of an operation (`A @ B`,
  `pc.matmul`, `ndarray @ A`, vector operands) goes through the same helper and the same
  engine entry function.
- **NumPy operands are imported, never computed with.** `numpy_import.hpp` turns an array
  into an engine matrix or vector; the engine computes.
- **Public or internal by name.** Every native name without a leading underscore is
  public: `pycauset.__all__` lists it and it needs a page under `documentation/docs/`
  (`tests/engine/test_api_surface.py` checks). Helpers the Python layer calls
  (`_sprinkle`, `_matrix_norm`, ...) start with an underscore.
- **No fallbacks.** If Python needs a native symbol, the build provides it; a missing
  symbol is an import error, not a silent alternative.

## Adding a binding

1. The engine function, per [[project/protocols/Adding Operations|Adding Operations]].
2. The binding in the subsystem's `bind_*.cpp`.
3. The public entry in `__init__.py` (or none, for an internal `_name`).
4. Tests in `tests/engine/`, an atlas entry for an operation, and the doc page.
