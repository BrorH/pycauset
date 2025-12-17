# NumPy Alignment Protocol

This protocol defines how user-facing Python names in `pycauset` should align with NumPy so the API is intuitive, predictable, and easy to learn.

---

## 1) Scope

This protocol applies to:

- Top-level Python module functions (factories, creators, convenience functions)
- Methods on user-facing array-like objects (vectors/matrices)
- Parameter names and return-shape conventions for those surfaces

This protocol does **not** force domain-specific objects (e.g. causal set / spacetime features) to mimic NumPy when the concepts do not exist in NumPy; those should still follow NumPy’s naming *style*.

---

## 2) Naming rules (public API)

### A) Functions / factories are lower-case

- Public factories and creators are lower-case and snake_case.
- If NumPy has the same concept with the same semantics, use the same name.

Examples:

- `pycauset.matrix(...)` (aligned with `np.array(...)` conceptually)
- `pycauset.vector(...)` (aligned with 1D `np.array([...])` output)
- `pycauset.zeros(...)`, `pycauset.ones(...)`, `pycauset.empty(...)`

### B) Concrete types remain PascalCase

- Native/optimized concrete classes remain PascalCase because they are types, not factories.

Examples:

- `pycauset.FloatMatrix`, `pycauset.Int8Vector`, `pycauset.TriangularBitMatrix`

### C) No deprecation aliases: rename == purge

When a public name changes:

- The old symbol is removed from the module surface.
- We do not ship “deprecated” wrappers or migration warnings.

Rationale: the project’s policy is purge-on-deprecate.

### D) Prefer NumPy parameter names and conventions

When applicable, match NumPy’s parameter naming and behavior:

- `dtype` for element type
- `shape` for dimensions
- `axis` and `keepdims` for reductions (future)

---

## 3) Behavioral alignment

### A) Shapes follow NumPy

- Matrices are 2D with `shape == (rows, cols)`.
- Vectors are 1D with `shape == (n,)`.

Constructor vs allocation rule:

- `pycauset.matrix(...)` and `pycauset.vector(...)` construct from data.
- Shape-based allocation uses `pycauset.zeros/ones/empty(shape, dtype=...)`.
- Scalars/0D arrays are out of scope; scalar input to `matrix/vector` must raise.

### B) `size()` and `shape` match NumPy

- `size()` is the total element count.
- `shape` exists as a property/attribute.
- If a callable form exists, `shape()` mirrors `shape`.

### C) `.fill(value)` exists

Provide a method analogous to NumPy’s `ndarray.fill(value)` on user-facing vector/matrix objects.

---

## 4) Canonical rename list (initial)

This protocol assumes a lower-case, NumPy-aligned public surface.

Canonical public entrypoints:

- `pycauset.matrix(...)` and `pycauset.vector(...)` construct from data.
- `pycauset.zeros/ones/empty(..., dtype=...)` allocate by shape.
- `pycauset.causal_matrix(...)` constructs causal matrices.
- `pycauset.causet(...)` is the convenience factory returning a `CausalSet`.

Policy:

- Public factories/functions are lower-case.
- PascalCase factory aliases are not part of the public API.

---

## 5) Documentation requirements (mandatory)

Any public rename/addition is not done until documentation is updated according to:

- `documentation/project/protocols/Documentation Protocol.md`

Minimum expectation:

- API reference pages exist for the final public names
- Examples match the final public names
- Cross-links updated (including roamlinks targets)

---

## See also

- `documentation/project/protocols/Documentation Protocol.md`
- `documentation/internals/plans/R1_SHAPES_PLAN.md`
