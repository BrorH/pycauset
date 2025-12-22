# Public API Contract

This page defines what counts as **public API** in PyCauset, what naming/import rules we follow, and how we handle removals.

This is intentionally strict: it keeps the project coherent while the internals evolve quickly.

---

## 1) What is public API?

Public API is anything a user is expected to import and rely on.

### Public (stable entrypoints)

- `pycauset.*` symbols that are documented in the API reference under `documentation/docs/`.
- The documented submodules:
  - `pycauset.spacetime`
  - `pycauset.field`
  - `pycauset.vis`

Rule of thumb:
- If it is meant for users, it must be reachable at `pycauset.*`.
- If it is not meant for users, it must live under a private namespace.

### Not public (may change anytime)

- Anything under `pycauset._internal.*`.
- Native binding details under `pycauset._pycauset`.
- Any symbol whose name starts with `_` (even if it is importable).

See also: [[dev/Python Internals|Python Internals]] and [[dev/Bindings & Dispatch|Bindings & Dispatch]].

---

## 2) Naming conventions

### Modules

- Public modules are short, nouns, and lower-case (examples: `spacetime`, `field`, `vis`).
- Internal modules live under `pycauset._internal`.

### Types (classes)

- Public types use `PascalCase`.
- Concrete matrix/vector classes follow `…Matrix` / `…Vector` naming.
  - Examples: `FloatMatrix`, `TriangularFloatMatrix`, `IntegerVector`.

### Matrix/vector semantic properties (`obj.properties`)

All public matrix/vector objects expose a `properties` mapping.

Contract:

- `obj.properties` is the single user-facing metadata container.
- Properties are **gospel**: they are not truth-validated by scanning payload bytes.
- Boolean-like properties use tri-state semantics via key presence:
  - unset means the key is absent,
  - explicit `False` is preserved.
- Some keys (e.g. `trace`, `determinant`) may be treated as **cached-derived** values:
  - they are persisted with validity signatures under `cached.*`,
  - and are surfaced back into `obj.properties` on load only when valid.

### Functions

- Public functions use `snake_case`.
  - Examples: `matmul`, `compute_k`, `invert`.

### Dtype tokens

PyCauset accepts NumPy-like dtype tokens as strings.

It also exports **dtype sentinel variables** on the top-level module for convenience (these are just strings).

Exported sentinels (examples):

- Integers: `pycauset.int8`, `pycauset.int16`, `pycauset.int32`, `pycauset.int64` (alias: `pycauset.int_`)
- Unsigned: `pycauset.uint8`, `pycauset.uint16`, `pycauset.uint32`, `pycauset.uint64` (alias: `pycauset.uint`)
- Floats: `pycauset.float16`, `pycauset.float32`, `pycauset.float64` (alias: `pycauset.float_`)
- Bit/boolean: `pycauset.bit`, `pycauset.bool_`
- Complex floats: `pycauset.complex_float16`, `pycauset.complex_float32`, `pycauset.complex_float64` (aliases: `pycauset.complex64`, `pycauset.complex128`)

Rule:
- Sentinel names follow NumPy conventions (including `_` suffix for built-in names like `bool_`).

- Float tokens: `"float16"`, `"float32"`, `"float64"` (aliases may exist: `"float"`, `"float_"`).
- Int tokens: `"int8"`, `"int16"`, `"int32"`, `"int64"` (aliases may exist: `"int"`, `"int_"`).
- Unsigned tokens: `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"` (alias: `"uint"`).
- Bit/boolean tokens: `"bit"`, `"bool"`, `"bool_"`.
- Complex float tokens: `"complex_float16"`, `"complex_float32"`, `"complex_float64"` (aliases: `"complex"`, `"complex64"`, `"complex128"`).

Rule:
- Tokens are lower-case, consistent with NumPy naming, and treated as part of the public surface.

### Exceptions and warnings

- Warnings derive from `PyCausetWarning` (examples: `PyCausetPerformanceWarning`, `PyCausetStorageWarning`).
- Exceptions should be specific and predictable; prefer raising built-in exceptions (`ValueError`, `TypeError`) unless a dedicated PyCauset exception is warranted.

See also: [[dev/Warnings & Exceptions|Warnings & Exceptions]].

### Global knobs / parameters

Some public configuration is intentionally exposed as module-level parameters.

- `pycauset.keep_temp_files` (documented under `documentation/docs/parameters/`)
- `pycauset.seed` (documented under `documentation/docs/parameters/`)

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
3. Update tests under `tests/python/`.
4. If native exports are involved, run the drift check:
   - `python tools/check_native_exports.py`

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
