# Public API Contract

This page defines what counts as **public API** in PyCauset, what naming/import rules we follow, and how we handle removals.

This is intentionally strict: it keeps the project coherent while the internals evolve quickly.

---

## 1) What is public API?

Public API is anything a user is expected to import and rely on.

### Public (stable entrypoints)

- `pycauset.*` symbols that are documented in the API reference under `documentation/docs/`.
- Public submodules in R1: `pycauset.spacetime` (the Minkowski diamond), `pycauset.synthetic`
  (poset generators) and `pycauset.vis` (lightweight plots). The rest of the physics layer
  (fields, other spacetimes) is parked in `experimental/physics`.

Rule of thumb:
- If it is meant for users, it must be reachable at `pycauset.*`.
- If it is not meant for users, it must live under a private namespace.

### Not public (may change anytime)

- Anything under `pycauset._internal.*`.
- Native binding details under `pycauset._pycauset`.
- Any symbol whose name starts with `_` (even if it is importable).

See also: [[internals/Python Internals|Python Internals]] and [[internals/Bindings & Dispatch|Bindings & Dispatch]].

---

## 2) Naming conventions

### Modules

- Public modules are short, nouns, and lower-case.
- Internal modules live under `pycauset._internal`.

### Types (classes)

- Public types use `PascalCase`.
- Concrete matrix/vector classes follow `…Matrix` / `…Vector` naming.
  - Examples: `DenseMatrix` (dtype `float64`), `TriangularMatrix` (dtype `float64`), `DenseVector` (dtype `int32`).

### Matrix/vector metadata

A matrix or vector carries its shape, dtype, structure (its class: `TriangularMatrix`,
`SymmetricMatrix`, `HermitianMatrix`, `DiagonalMatrix`, ...) and view state (scale factor,
transposition, conjugation), all saved with it. There is no user-assertion or notes store:
structure that operations exploit is the class, which guarantees it. Scalar results computed
from a matrix (`determinant`, `trace`, `sum`, `norm`, `matrix_rank`, `eigvals`, ...) are kept
with its storage until it changes, and saved with it.

### Functions

- Public functions use `snake_case`.
  - Examples: `matmul`, `compute_k`, `inv`.

### Dtype tokens

PyCauset accepts NumPy-like dtype tokens as strings.

It also exports **dtype sentinel variables** on the top-level module for convenience (these are just strings).

Exported sentinels (examples):

- Integers: `pycauset.int8`, `pycauset.int16`, `pycauset.int32`, `pycauset.int64` (`pycauset.int_` is NumPy's, int64)
- Unsigned: `pycauset.uint8`, `pycauset.uint16`, `pycauset.uint32`, `pycauset.uint64`
- Floats: `pycauset.float16`, `pycauset.float32`, `pycauset.float64`
- Bit/boolean: `pycauset.bit`, `pycauset.bool_`
- Complex floats: `pycauset.complex_float16`, `pycauset.complex_float32`, `pycauset.complex_float64` (aliases: `pycauset.complex64`, `pycauset.complex128`)

Rule:
- Sentinel names follow NumPy conventions (including `_` suffix for built-in names like `bool_`).

- Names: `"int8"` ... `"int64"`, `"uint8"` ... `"uint64"`, `"float16"`, `"float32"`,
  `"float64"`, `"bool"`, `"complex_float16"`, `"complex_float32"`, `"complex_float64"`
  (any case).
- PyCauset aliases: `"int"` (int32, PyCauset's default integer), `"uint"` (uint32), `"float"`,
  `"double"` (float64), `"single"` (float32), `"half"` (float16), `"bit"` / `"bool_"` (bool),
  `"complex"` (complex_float64), `"complex16"` (complex_float16).
- Everything NumPy accepts as a dtype: NumPy dtypes and scalar types (`np.float32`), names
  (`"complex64"`, `"complex128"`) and type codes, which mean what they mean in NumPy
  (`"i8"` is int64, `"f4"` float32, `"<u2"` uint16).
- Types PyCauset does not store (`np.longdouble`, `np.clongdouble`, strings, objects) raise.

### Exceptions and warnings

- Warnings derive from `PyCausetWarning` (examples: `PyCausetPerformanceWarning`, `PyCausetStorageWarning`).
- Exceptions should be specific and predictable; prefer raising built-in exceptions (`ValueError`, `TypeError`) unless a dedicated PyCauset exception is warranted.

See also: [[internals/Warnings & Exceptions|Warnings & Exceptions]].

### Global knobs / parameters

Some public configuration is intentionally exposed as module-level parameters.

- `pycauset.debug_mode` (documented under `documentation/docs/parameters/`)

---

## 3) Public vs internal boundaries (enforced by structure)

The intended layering:

- `python/pycauset/__init__.py` is the **public facade**.
- `python/pycauset/_internal/` holds implementation modules.
- `src/` + native bindings provide the engine.

If a new feature requires helpers, put helpers in `_internal/` and only re-export the intended entrypoint at `pycauset.*`.

---

## 4) Removals policy ("Deprecation" = purge)

PyCauset uses a purge policy:

- If we decide something should be removed, we remove it fully.
- We do not keep deprecated aliases or “deprecated but still present” codepaths.
- Documentation should never say “deprecated”: it should reflect the current reality.

If a removal breaks internal callers, update them in the same change.

---

## 5) How to change the public API safely

Changes to public API (names/semantics/import paths) are allowed pre-alpha, but must be deliberate.

Checklist:

1. Update the API reference page(s) under `documentation/docs/`.
2. Update any affected guides under `documentation/guides/`.
3. Update tests under `tests/engine/`.
4. Every public name needs its page: `tests/engine/test_api_surface.py` fails otherwise.

Related protocols:

- [[project/protocols/Pre-alpha Policy|Pre-alpha Policy]]
- [[project/protocols/Documentation Protocol|Documentation Protocol]]
- [[project/protocols/Adding Operations|Adding Operations]]
- [[project/protocols/Adding Types|Adding Types]]

---

## 6) When in doubt

If you’re unsure whether something is public:

- Default to *not public*.
- Keep it under `_internal/`.
- Only promote it to `pycauset.*` after it has a clear contract and an API reference page.
