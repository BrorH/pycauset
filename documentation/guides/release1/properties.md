# R1 Properties (Semantic Metadata)

Release 1 introduces `obj.properties`: a typed mapping that stores **semantic assertions** and **cached-derived values**.

Properties are a power-user feature. They can change which algorithms run, without scanning payload data.

!!! warning "Power-user semantics (gospel)"
  Properties are **authoritative assertions**. If you mark a matrix as diagonal/triangular/unitary/etc, PyCauset is allowed to run algorithms that assume the property is true.
  PyCauset does not validate truth by scanning payload.

## What shipped (R1)

R1_PROPERTIES is implemented end-to-end. In practice, that means:

- `obj.properties` exists on matrices and vectors and round-trips through `.pycauset`.
- Compatibility checks reject only **structurally impossible / contradictory** states (and do so in $O(1)$).
- Property propagation under metadata-only transforms (`transpose`, conjugation, scalar scale) is deterministic and $O(1)$.
- Cached-derived values (e.g. `trace`, `determinant`, `eigenvalues`) are validity-checked and invalidated on mutation.
- Core endpoints consult properties for fast paths (see “How properties affect algorithms”).

## What `properties` is

- `obj.properties` is a mapping (`str` keys → typed values).
- Many keys are boolean-like, but use **tri-state** behavior:
  - **unset**: the key is absent
  - **True**: asserted
  - **False**: explicitly negated

Most importantly:

- **Gospel assertions are authoritative**: PyCauset does not verify them by scanning data.
- **Compatibility checks are minimal and O(1)**: structurally impossible states raise immediately (e.g. `is_unitary=True` on a non-square matrix).

This design keeps the project scale-first: there is no hidden full-data pass to “validate” a claim.

## Common keys (Release 1)

### Gospel (semantic) assertions

These keys are treated as **semantic structure claims** (no truth validation):

- `is_zero`, `is_identity`, `is_permutation`
- `is_diagonal`, `is_upper_triangular`, `is_lower_triangular`
- `has_unit_diagonal`, `has_zero_diagonal`, `diagonal_value`
- `is_symmetric`, `is_anti_symmetric`, `is_hermitian`, `is_skew_hermitian`, `is_unitary`, `is_atomic`
- Vector-ish hints: `is_sorted`, `is_strictly_sorted`, `is_unit_norm`

### Cached-derived values

These keys are **derived/cached quantities** whose use is guarded by a validity signature (and which are invalidated on payload mutation):

- Persisted in `.pycauset` today: `trace`, `determinant`, `norm`, `sum`, `eigenvalues`
- Other cached keys may exist in-memory, but are not guaranteed to round-trip.

## Minimal example (triangular solve)

```python
import pycauset as pc

A = pc.identity(3)
b = pc.vector((1.0, 2.0, 3.0))

# Gospel assertion: solver is allowed to treat off-triangle entries as zero.
A.properties["is_upper_triangular"] = True

x = pc.solve_triangular(A, b)
```

## Setting, unsetting, and explicit False

Unset a key by deleting it, or by assigning `None`:

```python
A.properties["is_upper_triangular"] = True

del A.properties["is_upper_triangular"]
# or:
A.properties["is_upper_triangular"] = None
```

Explicit `False` is different from “unset”:

```python
A.properties["is_hermitian"] = False  # an explicit negation

## O(1) mutation semantics (effect summaries)

When payload is mutated, PyCauset updates/invalidates properties and cached-derived values without any extra pass over the data.

Internally this is handled by an $O(1)$ “health check” step that may consume a constant-size **effect summary** emitted by a kernel or operation.
Examples of effect bits include:

- “wrote any off-diagonal nonzero” (and whether it was above/below the diagonal)
- “wrote any diagonal entry” (and the last written diagonal value)
- “known all-zero” or “set identity” (for operations like explicit fills)

The important user-visible guarantee is: property/caching correctness does not require a second scan.

## Propagation under metadata-only transforms

Property propagation is deterministic and preserves tri-state semantics (unset stays unset; incompatible claims are unset rather than forced to `False`).

Key examples:

- Transpose swaps `is_upper_triangular` ↔ `is_lower_triangular`.
- Conjugation conjugates `diagonal_value` (when present) and cached complex values (e.g. `trace` / `determinant` / `eigenvalues`).
- Scalar scale updates cached-derived values when there is a safe $O(1)$ rule (e.g. `trace *= s`, `determinant *= s^n` when square) and clears them otherwise.
```

## How properties affect algorithms (Release 1 scope)

A few key endpoints are property-aware in R1:

- [[docs/functions/pycauset.solve.md|pycauset.solve]]
  - short-circuits `is_identity`
  - rejects `is_zero`
  - routes diagonal/triangular claims to `solve_triangular`

- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
  - may exploit `is_diagonal` and triangular claims for dispatch (without validating truth)

- [[docs/functions/pycauset.eigvalsh.md|pycauset.eigvalsh]]
  - consults/seeds cached `eigenvalues`
  - rejects if `is_hermitian` is explicitly `False`

## Cached-derived values (and why they’re different)

Some entries surfaced through `obj.properties` are **cached-derived** values (examples: `trace`, `determinant`, `norm`, `sum`, `eigenvalues`).

Rules:

- Cached-derived values are used only when their validity signature matches the current payload + view-state.
- Payload mutation clears or invalidates cached-derived values.
- Persistence stores cached-derived values separately (`cached.*`) and restores them only when valid.

Validity signatures (R1): cached entries are keyed to a payload identity and the view-state signature (transpose/conjugation/scalar).

The canonical snapshot/caching semantics are documented here:

- [[guides/Storage and Memory.md|Storage and Memory]]

## Persistence

Properties round-trip through `.pycauset` files:

- `properties.*` stores gospel assertions (including “missing vs explicit False”)
- `cached.*` stores cached-derived values with validity metadata

See [[docs/functions/pycauset.save.md|pycauset.save]] and [[docs/functions/pycauset.load.md|pycauset.load]].

## See also

- [[guides/Storage and Memory.md|Storage and Memory]]
- [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
- [[docs/classes/vector/pycauset.VectorBase.md|pycauset.VectorBase]]
- [[docs/functions/pycauset.solve.md|pycauset.solve]]
- [[docs/functions/pycauset.solve_triangular.md|pycauset.solve_triangular]]
- [[internals/DType System.md|DType System]]
- [[internals/plans/completed/R1_PROPERTIES_PLAN.md|R1_PROPERTIES plan artifact]]
