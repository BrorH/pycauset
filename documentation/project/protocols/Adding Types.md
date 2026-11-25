# Protocol: Adding a Structure or a dtype

**Status:** authoritative (2026 review, N.5). Companion to
[[project/protocols/Adding Operations|Adding Operations]].

**Objective:** a new (structure, dtype) combination that works in *every* operation, view and
storage mode the moment it exists, proven by the atlas.

PyCauset's value is specialised storage for every occasion (bit-packed causal matrices,
packed triangles, Hermitian matrices, …) in every dtype. Which combinations exist used to be
decided in five hand-kept tables that drifted apart (factory, clone, dispatch, load,
result-structure resolver); the gaps (no float16 diagonal, no complex triangular) came from
there. Now one catalogue decides.

## 1. The catalogue: one place decides

`include/pycauset/core/DTypeDispatch.hpp`:

| Piece | Role |
|---|---|
| `offered<Structure, T>()` | **the** predicate: does the pair exist? |
| `ClassOf<Structure, T>::type` | the C++ class holding the pair |
| `make_matrix` / `open_matrix` / `adopt_matrix` | create, load (persistence), clone: built from the two above |
| `visit_matrix(m, f)` | runtime dispatch to the concrete class: built from the two above |
| `throw_unsupported(s, dt)` | the error for a pair that does not exist, with the reason |

`ObjectFactory`, persistence (`open_storage`), the constructor bindings, `visit_matrix`, the
typed readers (`math/ViewRead.cpp`) and the result-structure resolver
(`core/StructureResolver.cpp`) all go through it. **Never add a `switch (dtype)` that lists
classes elsewhere.**

## 2. Adding a dtype to an existing structure

1. Make the class template work for the element type `T` (compile errors show what is
   missing). Typical work: accessors must use `view_value` (`math/ViewValue.hpp`) instead of
   `static_cast<double>`; complex `T` throws from `element_as_double` like `DenseMatrix`.
2. Flip `offered<S, T>()`.
3. Run the atlas (`python -m tests.atlas.report`): the new cells change from "unavailable"
   to live; fix every ✗ at its root (§5).
4. Update `tests/engine/test_catalogue.py` (`offered` there is the catalogue as data) and the
   class's API page (dtype list).

## 3. Adding a structure

1. `Structure` enum value (`core/Types.hpp`) and its name in `structure_name` /
   `parse_structure` (`DTypeDispatch.hpp`; the name is the persistence and Python key).
2. The class template, with the three constructor shapes every class has: create
   `(shape, file[, options])`, open `(shape, file, offset, seed, scalar, transposed[, options])`,
   adopt `(shape, mapper[, options])`; a typed `T get(i, j)` (honouring transposition) and
   `set(i, j, T)`; `clone()` keeping every view flag and option; accessors through
   `view_value`; `multiply_scalar` through `MatrixBase::scaled_clone` (O(1), keeps the view).
3. If one storage serves several structures, make the structure the option (the symmetric
   family: one `SymmetricMatrix<T>` with a kind for symmetric, antisymmetric, Hermitian and
   skew-Hermitian, each its own Python class through a marker base class).
4. Catalogue: `ClassOf` specialisation, `offered` rule, a branch in `detail::build_matrix` if
   the constructors take an option; `MatrixOptions` carries options so none is dropped
   (a dropped `has_diagonal` read shifted values twice: F-064, F-076).
5. Python: a `py::class_` for the structure's base (`bind_matrix.cpp`), its case in
   `structure_class_of` (`bindings_common.hpp`), the export in `pycauset/__init__.py`.
6. Persistence: the payload layout name in `persistence.py` (`_PAYLOAD_LAYOUT`).
7. Atlas: the structure in `STRUCTURES`, its NumPy reference in `reference()` and its
   construction in `build()` (`tests/atlas/atlas.py`); the timing harness's `_build_fast`.
8. Structure rules in operations (e.g. Hermitian + Hermitian stays Hermitian) are optional,
   per operation: see Adding Operations §3. Without them results are correct and dense.
9. Docs: class page in `documentation/docs/classes/matrix/`, the class index table,
   `CHANGELOG.md`.

## 4. Semantics to decide and write down

- The value of every element, including the mirrored ones (`conj`, sign) and the diagonal
  rule (zero, real, imaginary), enforced on `set`.
- `A.T` as an O(1) view where possible, derived from the view rule `conj?(raw · s)`: e.g.
  Hermitian `A.T = conj(A)` toggles the conjugation flag *and* conjugates the factor.
- Signed zeros: mirrored elements use IEEE negation (a stored 0 mirrors to -0); the atlas
  reference must build them by placement, not by subtraction.
- Which dtypes are meaningful (antisymmetric needs negation; Hermitian needs complex).

## 5. The gates

| Gate | Requirement |
|---|---|
| `pytest tests/atlas` | 0 ✗ over every operation × view × storage for the new combinations |
| `pytest tests/engine/test_catalogue.py` | the new pairs construct, save/load as themselves; rules hold |
| Both suites (`PYCAUSET_LAZY=0` too) | pass |

Bugs the new combinations expose in existing code are fixed at the root, never worked around
per type (N.5 found: a scaled view dropping the conjugation flag in three classes →
`scaled_clone`; identity shortcuts ignoring conjugation → `view_value(1, s, conj)`; complex
division by a signed zero → NumPy's algorithm in `scalar::true_divide`).

## See also

- [[project/protocols/Adding Operations|Adding Operations]]
- `include/pycauset/math/ViewValue.hpp`: the view rule
- `review/features/ATLAS.md`: the atlas
