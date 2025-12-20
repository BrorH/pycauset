# R1_FLAGS — Property Flags + Flag-Aware Algebra (Release 1)

**Status:** Draft (plan only; do not execute yet)

**Last updated:** 2025-12-20

## Purpose

Release 1 needs a **persistent property-flag system** that algorithms can rely on to select fast paths and specialized behavior.

These flags are **authoritative (“gospel”)**:

- The system does **not** validate that flagged properties are mathematically true.
- Implementations are allowed to behave *as if the property is true*.
- The only checks performed are **minimal compatibility/sanity checks** that prevent self-contradictory flag states or structurally impossible states (e.g., `is_unitary` on a non-square matrix).

This plan defines:

- the canonical flag schema,
- compatibility rules,
- propagation rules across metadata-only transforms,
- and where/how flags affect operator implementations and dispatch.

## Key references

- Roadmap node: `documentation/internals/plans/TODO.md` → **R1_FLAGS**
- Compute model: `documentation/internals/Compute Architecture.md`
- Persistence model: `python/pycauset/_internal/persistence.py` (and `.pycauset` metadata schema)
- Support readiness tracking: `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md`

## Constraints (non-negotiable)

- **No truth validation:** never scan matrix data to “confirm” a flag.
- **No backwards-compat aliases:** the public name is `flags` (not `user_flags`), and the persistence schema must not accept legacy names.
- **Deterministic behavior:** given the same flags and operation sequence, results and chosen algorithmic paths must be deterministic.
- **Minimal incompatibility checks only:** only reject flag combinations that are structurally impossible or internally contradictory for the flag lattice.
- **Propagation is mandatory:** metadata-only transforms must update flags predictably.
- **Testing + documentation phase at the end:** the final phase is an explicit “tests + docs” closure phase.

## Definitions

### Canonical representation

Public API concept: `obj.flags` is a mapping `dict[str, bool]`.

- Keys are stable, snake_case strings.
- Values are booleans (no tri-state and no missing fields once R1_FLAGS is complete).
- Flags are persisted in `.pycauset` metadata.

Internally, we may store flags as:

- a dense bitset keyed by a fixed enum, or
- a compact string-key map.

The public contract is the key set and behavior, not the internal storage.

### Flags included in R1_FLAGS (initial set)

This plan covers flags for **matrices and vectors**.

Matrix flags:

Special-case algebra (short-circuits):

- `is_zero`: treat every entry as $0$ (a “zero matrix” flag; not a shape claim).
- `is_identity`: treat as the identity matrix $I$.
- `is_permutation`: treat as a permutation matrix $P$ (re-indexing / row/col permutation).

Structure flags (index skipping / structured algorithms):

- `is_diagonal`
- `has_unit_diagonal`
- `is_upper_triangular`
- `is_lower_triangular`
- `is_strictly_upper_triangular`
- `is_strictly_lower_triangular`

Symmetry family:

- `is_symmetric`
- `is_anti_symmetric` (a.k.a. “skew-symmetric”)
- `is_hermitian`

Norm-preserving / spectral hints:

- `is_unitary`

Matrix-function hint:

- `is_atomic`: treat as an “atomic” triangular matrix in the sense used by common matrix-function algorithms (e.g. clustered diagonal values). This is gospel and is not validated.

Vector flags:

- `is_zero`: treat every entry as $0$ (a “zero vector” flag; not a shape claim).
- `is_unit_norm`: treat $\|v\|_2 = 1$.
- `is_sorted`
- `is_strictly_sorted`

Notes:

- “Strict” triangular excludes the diagonal.
- We do **not** add a broad zoo (SPD/PSD/normal/etc) in R1_FLAGS; those can be R2+.

## Compatibility model (minimal sanity checks)

These checks are allowed because they protect the internal semantics of the flag lattice and avoid impossible propagation states. They are **not** truth validation.

### Shape constraints

- `is_unitary=True` requires a square matrix.
- `is_hermitian=True` requires a square matrix.

- `is_identity=True` requires a square matrix.
- `is_permutation=True` requires a square matrix.
- `is_symmetric=True` requires a square matrix.
- `is_anti_symmetric=True` requires a square matrix.

For triangular-only flags:

- `is_upper_triangular=True` and/or `is_lower_triangular=True` requires a square matrix.
- `is_atomic=True` requires a square matrix.

### Lattice/implication constraints

Strict triangular implies non-strict triangular:

- If `is_strictly_upper_triangular=True` then `is_upper_triangular` must be True.
- If `is_strictly_lower_triangular=True` then `is_lower_triangular` must be True.

Diagonal consistency:

- If `is_diagonal=True` then:
  - `is_strictly_upper_triangular` must be False
  - `is_strictly_lower_triangular` must be False

Contradictions that must be rejected:

- `is_strictly_upper_triangular=True` together with `is_strictly_lower_triangular=True` (no non-zero matrix can satisfy both, and it makes the priority/dispatch ambiguous).

Ordering constraints:

- If `is_strictly_sorted=True` then `is_sorted` must be True.

### What is explicitly *not* validated

- Hermitian/unitary truth is never checked.
- “Hermitian implies diagonal implies triangular” is **not** auto-enforced.
- Setting flags that are mathematically inconsistent but not structurally contradictory is allowed.

## Priority tree (effective structure)

Operators that can exploit structure compute an **effective structure category** from flags without mutating the stored flags.

Priority (highest first):

0) Zero: if `is_zero=True`.
1) Identity: if `is_identity=True`.
2) Diagonal: treat as diagonal if `is_diagonal=True` OR if both `is_upper_triangular=True` and `is_lower_triangular=True`.
3) Strictly upper: if `is_strictly_upper_triangular=True`.
4) Strictly lower: if `is_strictly_lower_triangular=True`.
5) Upper triangular: if `is_upper_triangular=True`.
6) Lower triangular: if `is_lower_triangular=True`.
7) General.

Rationale:

- Diagonal dominates triangular for index skipping.
- “Both upper and lower” implies diagonal behavior for algorithms (even if the user did not set `is_diagonal`).

## Propagation rules

Flags must be transformed by metadata-only operations deterministically.

### Transpose

On transpose:

- `is_upper_triangular` ⇄ `is_lower_triangular`
- `is_strictly_upper_triangular` ⇄ `is_strictly_lower_triangular`
- `is_diagonal` stays the same
- `has_unit_diagonal` stays the same

- `is_zero` stays the same
- `is_identity` stays the same
- `is_permutation` stays the same

- `is_symmetric` stays the same
- `is_anti_symmetric` stays the same
- `is_hermitian` stays the same (if a matrix is Hermitian, its transpose is Hermitian)

- `is_atomic` stays the same

For `is_unitary`:

- `is_unitary` becomes False (transpose does not preserve unitarity in general)

Note: the final line is intentional: we do not attempt to “correct” user-set flags via propagation.

### Conjugation

On elementwise conjugation:

- Triangular/diagonal flags unchanged
- `is_hermitian` stays the same (if a matrix is Hermitian, its conjugate is Hermitian)

- `is_zero` stays the same
- `is_identity` stays the same
- `is_permutation` stays the same

- `is_symmetric` stays the same
- `is_anti_symmetric` stays the same

- `is_atomic` stays the same

For `is_unitary`:

- `is_unitary` becomes False (conjugation does not preserve unitarity in general)

### Adjoint (conjugate-transpose)

If an adjoint operation exists as a single transform:

- apply transpose rules plus conjugation rules, except:
  - `is_unitary` stays the same (adjoint preserves unitarity)

### Scalar multiply

On multiply-by-scalar:

- Triangular/diagonal flags unchanged

- `is_zero` stays the same

For `is_identity` and `is_permutation`:

- if the scalar is exactly $1$, keep the flag
- otherwise, the flag becomes False

For `is_hermitian`:

- if the scalar is real, `is_hermitian` stays the same
- otherwise, `is_hermitian` becomes False

For `is_unitary`:

- if $|scalar| = 1$, `is_unitary` stays the same
- otherwise, `is_unitary` becomes False

We do not infer *new* truths from the scalar (e.g., `scalar == 0` does not cause us to set additional flags to True).

### Views / clones

- Pure metadata-only views must copy flags into the new object (independent ownership), and then apply the propagation rules for the view transform.
- Clones/materializations preserve flags by value.

## Where flags affect algorithms (Release 1 scope)

Flags can legally change algorithm behavior. In R1_FLAGS we focus on the cases that create immediate leverage without broad rewrites.

Target operator families:

- Index skipping for structural loops (diagonal/triangular).
- Specialized solve paths:
  - triangular solve shortcuts when triangular flags are set.
- Specialized multiplications:
  - diagonal @ matrix and matrix @ diagonal.
- Spectral shortcuts:
  - `is_hermitian` chooses Hermitian eigen routines when available.
  - `is_unitary` allows using unitary identities where implemented.

The exact list of operators to update is tracked in SRP (op × dtype × device × structure table).

## Phases

### Phase A — Lock schema + compatibility rules

Deliverables:

- Canonical key set (initial flags) is finalized.
- Compatibility rules are finalized (shape constraints + lattice contradictions).
- Priority tree is finalized.

### Phase B — Public API surface contract

Deliverables:

- Define `flags` exposure shape in Python (`dict[str,bool]` with stable keys).
- Define mutation contract (set whole mapping vs per-key updates) and error behavior for invalid combinations.

### Phase C — Persistence + metadata schema

Deliverables:

- Define `.pycauset` `metadata.json` fields required to store flags.
- Define the rule: after R1_FLAGS, all persisted objects must include the canonical flag set.

### Phase D — Propagation integration

Deliverables:

- Ensure metadata-only transforms update flags per the propagation rules.
- Ensure clones/materializations preserve flags.

### Phase E — Flag-aware operator wiring

Deliverables:

- Identify the minimal operator set that must become flag-aware in R1 (tracked via SRP inventory).
- Implement algorithm selection using the priority tree (effective structure), without mutating stored flags.

### Phase F — Tests + documentation (FINAL)

Deliverables:

- Tests:
  - compatibility checks (invalid combinations reject deterministically),
  - propagation checks (transpose/conjugate/scalar multiply),
  - persistence round-trip preserves flags.
- Docs:
  - user-facing docs for `flags` (semantics: gospel, not validated),
  - operator docs where flags change behavior.

## Open questions (must be resolved before Phase B)

- Do we want to keep `is_unitary` propagation rules as written (transpose/conjugate clear; adjoint preserves), or should `is_unitary` be preserved more aggressively?
- Should vectors have flags in R1_FLAGS, or only matrices? (This plan assumes matrices only; vectors can be added later if needed.)
