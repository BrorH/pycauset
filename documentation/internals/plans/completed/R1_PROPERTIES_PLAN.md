# R1_PROPERTIES — Semantic Properties + Property-Aware Algebra (Release 1)

**Status:** Phase A–F implemented (Phase E expanded: property-aware matmul/solve/eigvalsh; O(1) mutation effect summaries integrated)

**Last updated:** 2025-12-21

> Documentation note:
>
> This file is a planning/spec artifact. User-visible R1 behavior is documented in:
>
> - [[guides/release1/properties.md|Release 1: Properties (what shipped)]]
> - [[guides/Storage and Memory.md|Storage and Memory]] (how properties/caches persist)
> - API footprint: [[docs/classes/matrix/pycauset.MatrixBase.md|MatrixBase]] and [[docs/classes/vector/pycauset.VectorBase.md|VectorBase]]

## Purpose

Release 1 needs a **semantic properties system** that algorithms can rely on to select fast paths, specialized behavior, and stable dispatch.

This is a deliberate shift from “structure-by-type” (e.g. selecting triangular behavior by checking whether an object is a triangular *class*) to **structure-by-properties**. In R1_PROPERTIES, algorithms treat the relevant properties as the authoritative source of truth.

These properties are **authoritative (“gospel”)**:

- The system does **not** validate that asserted semantic properties are mathematically true.
- Implementations are allowed to behave *as if the property is true*.
- The only checks performed are **minimal compatibility/sanity checks** that prevent self-contradictory property states or structurally impossible states (e.g., `is_unitary` on a non-square matrix).

This plan defines:

- the canonical properties schema,
- compatibility rules,
- propagation rules across metadata-only transforms,
- the derived-cache correctness/invalidation model,
- and where/how properties affect operator implementations and dispatch.

## At a glance (the contract)

- **Properties:** `obj.properties` is a typed mapping exposed to users.
- **Hard errors on incompatibility:** structurally impossible / internally contradictory asserted states raise immediately.
- **No scans:** neither property rules nor cache validation require scanning payload.
- **Cached derived values:** some entries in `obj.properties` are cached derived values (e.g., `trace`) with strict validity rules; they are never treated as semantic structure properties.
- **Health check is $O(1)$:** no additional payload passes; parallel kernels may emit a constant-size effect summary.
- **Storage integration:** see [[guides/Storage and Memory]] (format + semantics). The container mechanics were tracked under the R1_STORAGE plan during implementation.

## Implementation status (as of 2025-12-21)

This plan is implemented end-to-end for Release 1 scope.

- **Properties mapping exposed:** `obj.properties` exists on native objects via Python patching (`python/pycauset/_internal/properties.py`, installed in `python/pycauset/__init__.py`).
- **Phase A compatibility enforced:** `obj.properties` is a validating mapping; structurally impossible / contradictory asserted states raise immediately (no payload scans). Compatibility now includes square-only structure flags (triangular/atomic), ordering constraints (strictly-sorted ⇒ sorted), and the “upper+lower ⇒ not diagonal-false” constraint.
- **Persistence bridging present:** `meta["properties"]` round-trips; cached-derived values persist under `cached.*` and are surfaced only when signatures match (`python/pycauset/_internal/persistence.py`).
- **Metadata-only view propagation present:** `transpose`, `conj`, `.H` propagate properties in O(1) (`python/pycauset/_internal/properties.py`).
- **Cached-derived propagation improved:** scalar multiply and add/sub apply the Phase-A propagation rules for cached-derived values where safe (O(1)), including `sum`.
- **Priority tree implemented:** an internal helper computes an effective structure category from properties (zero/identity/diagonal/triangular/general) without scanning payload.
- **Operator wiring (Phase E expanded):** property-aware routing exists for `matmul`, `solve`/`solve_triangular`, and `eigvalsh` (cached eigenvalues). `trace()`/`determinant()` and scalar-returning `pycauset.norm()`/`pycauset.sum()` continue to consult/seed cached-derived values.
- **Tests + docs exist:** see `tests/python/test_properties_persistence.py` and `tests/python/test_properties_operator_wiring.py`; docs updated in the storage/semantics guides.

Open gaps relative to the plan:

- None for R1. Kernel-style effect summaries are applied via the properties shim (`_apply_effect_summary_inplace`) and are exercised by mutation patches; device kernels can emit the same constant-size summaries later without changing semantics.

## Key references

- Roadmap node: `documentation/internals/plans/TODO.md` → **R1_PROPERTIES**
- Compute model: `documentation/internals/Compute Architecture.md`
- Storage semantics + format (canonical docs): `documentation/guides/Storage and Memory.md`
- Current persistence implementation: `python/pycauset/_internal/persistence.py`
- Support readiness tracking: `documentation/internals/plans/SUPPORT_READINESS_FRAMEWORK.md`

## Constraints (non-negotiable)

- **No truth validation:** never scan matrix/vector data to “confirm” a property.
- **No backwards-compat aliases:** the public name is `properties` (not `flags`, not `user_flags`), and the persistence schema must not accept legacy names.
- **Deterministic behavior:** given the same properties and operation sequence, results and chosen algorithmic paths must be deterministic.
- **Minimal incompatibility checks only:** only reject property combinations that are structurally impossible or internally contradictory for the property lattice.
- **Propagation is mandatory:** metadata-only transforms must update properties predictably.
- **Testing + documentation phase at the end:** the final phase is an explicit “tests + docs” closure phase.

## Design anchors (agreed)

These are design principles specific to R1_PROPERTIES (in addition to the project-wide mantras).

- **Properties-as-gospel (power-user feature):** properties are authoritative claims. If a matrix is marked as triangular, triangular-aware algorithms are allowed to behave as if entries outside the relevant triangle are zero, even if the underlying payload is not triangular.
- **No truth validation:** the system must never scan matrix/vector data to “confirm” a property.
- **Lazy-evaluation invariant (metadata scaling):** property compatibility checks, property propagation, and effective-structure selection must be $O(1)$ and must depend only on existing metadata (shape, dtype, existing properties) plus the transform parameters (e.g. the scalar value).
- **Bedrock scope:** every `PersistentObject` must have properties support. R1 includes matrices and vectors.
- **Documentation is part of the feature:** user-facing docs must explicitly teach the “power user” semantics and the fact that properties can change algorithm choices.
- **Hybrid caching (agreed):** some expensive derived quantities should be preserved under certain transforms using only $O(1)$ rules; otherwise they must be unset. No cache exists if it forces recomputation.

## Definitions

### Terminology

- “Property” means a typed metadata item attached to a `PersistentObject`.
- `obj.properties` is the single user-facing mapping, but it contains two semantic classes of entries:
  - **Gospel assertions:** structural/special-case intent (triangular/diagonal/unitary/etc). Never truth-validated.
  - **Cached-derived values:** `trace`/`determinant`/`rank`/`norm` and similar. Validity-checked and may be cleared.
- Some keys are boolean-like; for those, we use a tri-state convention via key presence.

### Canonical representation

Public API concept: `obj.properties` is a mapping from stable string keys to **typed values**.

- Keys are stable, snake_case strings.
- Values are typed and may be **unset**.
- Properties are persisted in `.pycauset` metadata (see “Persistence + metadata schema”).

### Value model (Release 1)

`obj.properties` is a typed mapping (string keys → typed values). For boolean-like keys we use tri-state semantics:

- `True`: asserted; algorithms may exploit it.
- `False`: explicitly negated.
- Unset: no claim.

Public API decision (agreed):

- Unset means the key is **absent** from `obj.properties`.
- Explicit `False` is represented by the key being present with value `False`.

Persistence and internal representations must preserve “missing vs explicit False”.

### Metadata taxonomy (aligns with R1_STORAGE)

R1_STORAGE defines three kinds of metadata on disk (identity/header vs view-state vs user-facing properties/caches). R1_PROPERTIES defines the **user-facing semantics** and how operators interpret and propagate them.

In practice:

1) **Identity/header metadata** (system-managed)
  - Shape/dtype and any payload layout descriptor needed to interpret bytes.
  - Not user-facing.

2) **View-state metadata** (system-managed)
  - Examples: transpose/conjugation/adjoint state, scalar factors.
  - Must be metadata-only and $O(1)$.
  - Participates in cached-derived validity signatures.

3) **User-facing `properties`** (single mapping; two semantic classes)
  - Exposed as `obj.properties`.
  - Gospel assertions are authoritative and never truth-validated.
  - Cached-derived values are user-facing via clean keys, but are persisted as caches with validity metadata (see Phase C).

### Caching system (Release 1)

R1 introduces a more rigorous caching model so that derived metadata remains correct under mutation and metadata-only transforms.

Principles:

- **Correctness first:** cached values must never be used if they might be stale.
- **No scans for validation:** cache validity checks must be $O(1)$ and based on metadata/versioning only.
- **Deterministic:** given the same history of operations, caches must be filled/cleared deterministically.

#### Post-operation property health check (required)

Every operation that returns a new `PersistentObject` and every in-place mutation must run a cheap, deterministic **post-operation health check** step that:

- applies property propagation rules for metadata-only transforms,
- updates/propagates cached derived values where safe ($O(1)$ rules only),
- clears any cached derived values that are no longer guaranteed to match the object’s semantics,
- performs minimal compatibility checks (no scans).

This “health check” must not require **any payload scan** or **any additional payload pass**.

Clarification (parallelization-friendly contract):

- Compute kernels (CPU/GPU/streaming) are always allowed to read/write payload as required to produce correct results.
- The health check is a **metadata normalization** step, not a compute step.
- The health check may use:
  - existing metadata (shape, dtype, view-state, existing properties), and
  - the operation parameters (e.g., transpose, scalar value, indices for `M[i, j] = ...`), and
  - an optional constant-size **effect summary** produced by the operation while it runs.

The key rule is: property/caching correctness must not force a second pass over payload.

Complexity requirement (agreed):

- The health check must be strictly **$O(1)$**.
- No $O(\log N)$ work is allowed here (even for “small” metadata updates), because this check must also run after very frequent primitives such as element `set()`.

### Effect summaries (enabler for parallel kernels)

To make future parallelization plug in cleanly (via `AutoSolver` / `ComputeDevice`) without special integration work, operations may emit an **effect summary**:

- A small fixed-size struct (conceptually) that answers a few yes/no questions about what the op *definitely* did.
- It is computed “for free” during the operation (e.g., via an OR-reduction of per-thread booleans), without extra passes.
- If an op cannot cheaply provide a fact, it reports “unknown” and the health check conservatively unsets affected properties/caches.

Examples of useful effect bits:

- “payload mutated” (increments content epoch)
- “wrote any off-diagonal element”
- “wrote any diagonal element”
- “result is known-all-zero” (only for ops like explicit zero-fill; not for general matmul)

This keeps the health check deterministic and $O(1)$, while letting highly-parallel kernels remain unconstrained.

Where the health check is required (non-exhaustive; must be completed in Phase A):

- After any **payload mutation** method (e.g., element `set`, bulk fill, random fill).
- After any **metadata-only transform** that changes view-state (transpose/conjugate/adjoint, scalar changes).
- After any operation that **constructs a view** (clones, slices/views, transpose views).
- After **load/deserialize** (properties and caches must be normalized).
- After any API that lets users set properties explicitly (power-user setter).

Minimum design requirements:

- Every `PersistentObject` maintains a cheap-to-check **content version** (e.g., a monotonically increasing epoch) that changes whenever payload data is mutated.
- Every cached-derived entry records the version/signature it depends on (payload version + relevant view-state signature).
- On lookup, the system uses a cached-derived value only if the dependency signature matches; otherwise it is ignored/cleared.

Cached-derived propagation under metadata-only transforms:

- Some caches can be preserved or transformed without scanning (e.g., trace and determinant under transpose; trace scales linearly with scalar; determinant scales by $scalar^n$ for $n\times n$).
- If a cache does not have a safe propagation rule, it must be cleared to avoid stale answers.

#### Core cached-derived propagation table (initial; must be completed in Phase A)

This is the minimum expected “cheap propagation” set for R1.

Definitions:

- “Unset” means the entry is removed from the cached-derived store.
- If a rule says “requires X known”, it means the input cache entry must be present.

| Operation / transform | `trace` | `determinant` | `rank` | `norm` (vector) |
|---|---|---|---|---|
| transpose | preserve (if known) | preserve (if known) | preserve (if known) | preserve (if known) |
| conjugation | conjugate (if known) | conjugate (if known) | preserve (if known) | preserve (if known) |
| adjoint | conjugate (if known) | conjugate (if known) | preserve (if known) | preserve (if known) |
| scalar multiply by $s$ | if known: `trace *= s` | if known and square: `det *= s^n` | if known: if $s=0$ then `rank=0` else preserve | if known: `norm *= |s|` |
| add/subtract | if both known: add/sub | unset | unset | unset |
| any payload mutation (e.g. set element) | unset | unset | unset | unset |
| any other op without an explicit cheap rule | unset | unset | unset | unset |

Notes:

- “payload mutation” means modifying stored data, not changing metadata-only view state.
- This table applies to cached-derived values (regardless of whether the value was produced by computation or set by a power user).

R1 decision (updated):

- `trace`, `determinant`, `rank`, and vector `norm` are cached-derived values, not semantic structure properties.
- There is **no semantic distinction** between “user-set” and “computed” cache values. If a user sets `trace=231.2`, that is simply the cached trace value.
- Public methods/endpoints (e.g., `trace()`) may use a cached value if present and valid; otherwise they compute and then populate the cache.

#### User-populated caches (power-user feature)

Power users may set derived cache values directly to avoid expensive computation.

Semantics (agreed by direction):

- A user-populated cache entry is treated exactly like a computed cache entry.
- Cache validity is still enforced: a cache entry is used only when its dependency signature matches (payload content epoch + relevant view-state signature).
- If the payload is mutated, the dependency signature changes and the cached value is ignored/cleared.
- If a metadata-only transform occurs and there is no explicit $O(1)$ propagation rule for that cache key, the cache entry is cleared.

This preserves the correctness bar (“never stale”) while still allowing power users to seed expensive derived quantities.

Provenance note:

- The system does not expose a user-visible distinction between “user-set” and “computed” cache values.
- Internal provenance tracking is optional (debug/telemetry only) and must not change semantics.

#### Robust general cache system (required; makes future cached quantities easy)

R1 must define a general cache system that supports adding cached quantities later without redesign.

Contract:

- Cached-derived values have a **cache store** (internal backing) with validity metadata.
- Cache keys are stable identifiers (e.g., `trace`, `determinant`, `rank`, `norm`).
- Each cache entry records:
  - a typed value, and
  - a dependency signature (at minimum: payload content epoch + view-state signature).
- Cache lookup is deterministic and $O(1)$.
- Cache validity checks and propagation never require scanning payload.

Registration/extension model:

- Each cache key must define:
  - its value type,
  - what it depends on (payload epoch + which view-state components matter), and
  - explicit $O(1)$ propagation rules for the supported metadata-only transforms.
- If a cache key has no rule for a given transform, the default behavior is to clear it.

Practical outcome:

- Adding a new cached quantity later is a matter of defining the key and its propagation/invalidation rules, not inventing a new ad-hoc persistence path or special-case logic inside operators.

### Properties included in R1_PROPERTIES (initial set)

This plan covers properties for **matrices and vectors**.

Matrix properties:

Special-case algebra (short-circuits):

- `is_zero`: treat every entry as $0$ (a “zero matrix” property; not a shape claim).
- `is_identity`: treat as an identity-like matrix $I$.
  - Important: PyCauset allows **rectangular identity** (see `pycauset.identity()`), so `is_identity=True` is allowed for non-square matrices.
- `is_permutation`: treat as a permutation matrix $P$ (re-indexing / row/col permutation).

Structure properties (index skipping / structured algorithms):

- `is_diagonal`
- `diagonal_value` (typed; if present, asserts every diagonal element equals this value)
- `has_unit_diagonal` (shorthand; equivalent to `diagonal_value = 1` when meaningful)
- `is_upper_triangular`
- `is_lower_triangular`
- `has_zero_diagonal` (shorthand; equivalent to `diagonal_value = 0` when meaningful)

Symmetry family:

- `is_symmetric`
- `is_anti_symmetric` (a.k.a. “skew-symmetric”)
- `is_hermitian`
- `is_skew_hermitian`

Norm-preserving / spectral hints:

- `is_unitary`

Matrix-function hint:

- `is_atomic`: treat as an “atomic” triangular matrix in the sense used by common matrix-function algorithms (e.g. clustered diagonal values). This is gospel and is not validated.

Cached-derived matrix properties (cache semantics; validity-checked; may be cleared):

- `trace`
- `determinant`
- `rank`
- `sum`

Vector properties:

- `is_zero`: treat every entry as $0$ (a “zero vector” property; not a shape claim).
- `is_unit_norm`: treat $\|v\|_2 = 1$.
- `is_sorted`
- `is_strictly_sorted`

Cached-derived vector properties (cache semantics; validity-checked; may be cleared):

- `norm` (e.g. cached $\|v\|_2$)
- `sum`

Notes:

- Strict triangular is represented as `diagonal_value=0` + triangular (no separate strict-triangular booleans).
- We do **not** add a broad zoo (SPD/PSD/normal/etc) in R1_PROPERTIES; those can be R2+.

Important: the public/user-facing concept is **properties**, and the system must be ready for non-boolean properties (e.g., numeric metadata like `rank`) without redesigning the plumbing.

## Compatibility model (minimal sanity checks)

These checks are allowed because they protect the internal semantics of the property lattice and avoid impossible propagation states. They are **not** truth validation.

Hard error policy (agreed):

- If a caller (user or internal) attempts to assert an **incompatible** property state (e.g. `is_unitary=True` on a non-square matrix), this must raise a **hard error immediately**.
- This is not “validation”; it is enforcing internal semantic consistency.

Tri-state scope rule:

- Compatibility checks primarily constrain **asserted structural truths** (keys with value `True`).
- Unset (missing) means “no claim” and does not create contradictions by itself.
- Explicit `False` must be preserved, but does not introduce additional obligations unless it directly contradicts a required implication of an asserted `True`.

### Shape constraints

- `is_unitary=True` requires a square matrix.
- `is_hermitian=True` requires a square matrix.

- `is_identity=True` is allowed for rectangular matrices (rectangular identity semantics).
- `is_permutation=True` requires a square matrix.
- `is_symmetric=True` requires a square matrix.
- `is_anti_symmetric=True` requires a square matrix.
- `is_skew_hermitian=True` requires a square matrix.

Diagonal note (agreed):

- `is_diagonal=True` is allowed for rectangular matrices. Semantics: all off-diagonal entries are treated as zero, with the diagonal defined for indices $i=i$ up to $\min(m,n)$.

For triangular-only properties:

- `is_upper_triangular=True` and/or `is_lower_triangular=True` requires a square matrix.
- `is_atomic=True` requires a square matrix.

Shape constraints apply only when the relevant property value is `True`.

### Lattice/implication constraints

Strict triangular note:

- R1 avoids a dedicated strict-triangular boolean zoo.
- Instead, “strictly upper/lower triangular” is represented as:
  - `is_upper_triangular=True` (or `is_lower_triangular=True`) **and**
  - `diagonal_value = 0` (or `has_zero_diagonal=True`).

Diagonal consistency:

If `is_diagonal=True` then it may still be compatible with `diagonal_value=0` (zero diagonal) or `diagonal_value=1` (unit diagonal).

Diagonal-value consistency (agreed):

- `diagonal_value` does not imply `is_diagonal`.
- If `has_unit_diagonal=True`, then `diagonal_value` (if present) must equal $1$.
- If `has_zero_diagonal=True`, then `diagonal_value` (if present) must equal $0$.

Identity consistency (agreed):

- `is_identity=True` requires:
  - `is_zero` is not `True`.
  - `is_diagonal` is not `False`.
  - `has_unit_diagonal` is not `False`.
  - `has_zero_diagonal` is not `True`.
  - `diagonal_value` (if present) must equal $1$.

Contradictions that must be rejected:

- `is_upper_triangular=True` together with `is_lower_triangular=True` is only meaningful as “diagonal-like” behavior; if the user also asserts `is_diagonal=False`, that must be rejected.

Symmetry-family sanity checks (minimal):

- `is_symmetric=True` and `is_anti_symmetric=True` is rejected unless `is_zero=True`.
- `is_hermitian=True` and `is_skew_hermitian=True` is rejected unless `is_zero=True`.

Ordering constraints:

- If `is_strictly_sorted=True` then `is_sorted` must be True.

Tri-state refinement:

- If `is_strictly_sorted=True` then `is_sorted` must not be `False` (it may be `True` or `None`).

### What is explicitly *not* validated

- Hermitian/unitary truth is never checked.
- “Hermitian implies diagonal implies triangular” is **not** auto-enforced.
- Setting properties that are mathematically inconsistent but not structurally contradictory is allowed.

## Priority tree (effective structure)

Operators that can exploit structure compute an **effective structure category** from properties without mutating stored properties.

Terminology note: this is computed from properties.

Implementation note (performance): the effective structure category must be computed once per operation and passed downward (do not repeatedly query the properties container inside inner loops).

Priority (highest first):

0) Zero: if `is_zero=True`.
1) Identity: if `is_identity=True`.
2) Diagonal: treat as diagonal if `is_diagonal=True` OR if both `is_upper_triangular=True` and `is_lower_triangular=True`.
3) Upper triangular: if `is_upper_triangular=True`.
4) Lower triangular: if `is_lower_triangular=True`.
5) General.

Rationale:

- Diagonal dominates triangular for index skipping.
- “Both upper and lower” implies diagonal behavior for algorithms (even if the user did not set `is_diagonal`).

## Propagation rules

Properties must be transformed by metadata-only operations deterministically.

Tri-state propagation rule:

- Propagation must preserve the distinction between `False` and `None`.
- When a transform does not preserve a property in general, the propagated value should become `None` (unset), not `False`.

### Transpose

On transpose:

- `is_upper_triangular` ⇄ `is_lower_triangular`
  - strictness is represented via `diagonal_value=0`, which is preserved under transpose.
- `is_diagonal` stays the same
- `has_unit_diagonal` stays the same
- `has_zero_diagonal` stays the same
- `diagonal_value` stays the same

- `is_zero` stays the same
- `is_identity` stays the same
- `is_permutation` stays the same

- `is_symmetric` stays the same
- `is_anti_symmetric` stays the same
- `is_hermitian` becomes None unless the dtype is real (transpose does not preserve Hermitian-ness in general)
- `is_skew_hermitian` becomes None unless the dtype is real (transpose does not preserve skew-Hermitian-ness in general)

- `is_atomic` stays the same

For `is_unitary`:

- `is_unitary` becomes None (transpose does not preserve unitarity in general)

Note: the final line is intentional: we do not attempt to “correct” user-set properties via propagation.

### Conjugation

On elementwise conjugation:

- Triangular/diagonal properties unchanged
- `is_hermitian` becomes None unless the dtype is real (conjugation does not preserve Hermitian-ness in general)
- `is_skew_hermitian` becomes None unless the dtype is real (conjugation does not preserve skew-Hermitian-ness in general)

- `diagonal_value` is conjugated (if present)

- `is_zero` stays the same
- `is_identity` stays the same
- `is_permutation` stays the same

- `is_symmetric` stays the same
- `is_anti_symmetric` stays the same

- `is_atomic` stays the same

For `is_unitary`:

- `is_unitary` becomes None (conjugation does not preserve unitarity in general)

### Adjoint (conjugate-transpose)

If an adjoint operation exists as a single transform:

- apply transpose rules plus conjugation rules, except:
  - `is_unitary` stays the same (adjoint preserves unitarity)
  - `is_hermitian` stays the same (adjoint preserves Hermitian-ness)
  - `is_skew_hermitian` stays the same (adjoint preserves skew-Hermitian-ness)

### Scalar multiply

On multiply-by-scalar:

- Triangular/diagonal structural properties unchanged

- `diagonal_value` scales by the scalar (if present)

- `is_zero` stays the same

For `is_identity` and `is_permutation`:

- if the scalar is exactly $1$, keep the property
- otherwise, the property becomes False

For `is_hermitian`:

- if the scalar is real, `is_hermitian` stays the same
- otherwise, `is_hermitian` becomes None

For `is_skew_hermitian`:

- if the scalar is real, `is_skew_hermitian` stays the same
- otherwise, `is_skew_hermitian` becomes None

For `is_unitary`:

- if $|scalar| = 1$, `is_unitary` stays the same
- otherwise, `is_unitary` becomes None

We allow propagation rules to invalidate properties (set to `None` or `False`) when the operation semantics make that deterministic, but we do not infer *new* `True` claims from the scalar (e.g., `scalar == 0` does not cause us to set additional properties to True).

### Views / clones

- Pure metadata-only views must copy properties into the new object (independent ownership), and then apply the propagation rules for the view transform.
- Clones/materializations preserve properties by value.

## Where properties affect algorithms (Release 1 scope)

Properties can legally change algorithm behavior. In R1_PROPERTIES we focus on the cases that create immediate leverage without broad rewrites.

Target operator families:

- Index skipping for structural loops (diagonal/triangular).
- Specialized solve paths:
  - triangular solve shortcuts when triangular properties are set.
- Specialized multiplications:
  - diagonal @ matrix and matrix @ diagonal.
- Spectral shortcuts:
  - `is_hermitian` chooses Hermitian eigen routines when available.
  - `is_unitary` allows using unitary identities where implemented.

The exact list of operators to update is tracked in SRP (op × dtype × device × structure table).

In particular, type-based structural dispatch (e.g., “triangular-by-class”) must be replaced (or gated) so that properties are the authoritative source of structural intent. Typed storage formats remain important for performance, but they are not the semantic authority.

## Phases

### Phase A — Lock schema + compatibility rules

**Status (R1): Implemented**

Implementation + coverage:

- Compatibility enforcement: `python/pycauset/_internal/properties.py` (`obj.properties` validating mapping)
- Tests: `tests/python/test_properties_compatibility.py`

Deliverables:

- Canonical key set (initial properties) is finalized.
- Compatibility rules are finalized (shape constraints + lattice contradictions).
- Priority tree is finalized.
- Tri-state semantics (`True`/`False`/`None`) and its propagation rules are finalized.
- Lazy-evaluation invariant is explicitly enforced (no propagation/compatibility rule may require scanning data).
- Caching model requirements are locked (content-versioning + safe propagation vs clear rules).

### Phase B — Public API surface contract

**Status (R1): Implemented**

Implementation + coverage:

- Public surface: `obj.properties` mapping with tri-state booleans via key presence
- Assignment + per-key updates: `python/pycauset/_internal/properties.py`

Deliverables:

- Define `properties` exposure shape in Python (mapping of stable keys to typed values; boolean-like keys use tri-state semantics via presence).
- Define mutation contract (set whole mapping vs per-key updates) and error behavior for invalid combinations.

### Phase C — Persistence + metadata schema

**Status (R1): Implemented**

Implementation + coverage:

- Persistence bridge: `python/pycauset/_internal/persistence.py`
- Tests: `tests/python/test_properties_persistence.py`

Deliverables:

- Define the `.pycauset` metadata schema required to store:
  - gospel `properties` (including missing-vs-explicit semantics), and
  - cached-derived values (validated by version/signature).
- Define the rule: persisted objects must preserve property values *and* whether each boolean-like property is explicitly set vs unset.

Naming/encoding note (must be decided in Phase C):

- User-facing keys are clean (e.g., `trace`), but persistence must also encode “this is cached-derived” + dependency signatures.
- Decision (agreed): store cached-derived values under a dedicated `cached` / `caches` section keyed by clean names (e.g., `cached.trace`).

Implications:

- In memory: users read/write `obj.properties["trace"]`.
- On disk: `metadata` stores cached-derived values under `cached.trace` (and friends), alongside the dependency signature.
- On load: if a cached-derived value is present and its dependency signature matches the restored object state, it is surfaced as `obj.properties["trace"]`.
- On save: cached-derived values are written under `cached.*` (never as top-level keys like `cached_trace`).

Storage-format note:

- The single-file container format and typed metadata encoding are tracked under `documentation/internals/plans/R1_STORAGE_PLAN.md`.
- R1_PROPERTIES must remain storage-format agnostic: the `properties` and derived-cache semantics must plug into the storage layer without changing frontend save/load call sites.

### Phase D — Propagation integration

**Status (R1): Implemented**

Implementation + coverage:

- View propagation: transpose/conj/adjoint in `python/pycauset/_internal/properties.py`
- Scalar + add/sub cached-derived propagation in `python/pycauset/_internal/properties.py`
- Tests: `tests/python/test_properties_propagation.py` and `tests/python/test_properties_compatibility.py`

Deliverables:

- Ensure metadata-only transforms update properties per the propagation rules.
- Ensure clones/materializations preserve properties.
- Ensure derived caches are invalidated or safely propagated under metadata-only transforms.

### Phase E — Property-aware operator wiring

**Status (R1): Implemented**

Implemented in R1:

- Priority tree helper for effective structure selection (no scans)
- Scalar operator wiring: `trace()`, `determinant()`, `pycauset.norm()`, `pycauset.sum()`
- Structured routing:
  - `matmul` routes diagonal/triangular cases based on effective structure.
  - `solve` routes diagonal/triangular cases via `solve_triangular`.
  - `eigvalsh` consults/seeds cached-derived `eigenvalues`.

Deliverables:

- Identify the minimal operator set that must become property-aware in R1 (tracked via SRP inventory).
- Implement algorithm selection using the priority tree (effective structure), without mutating stored properties.

### Phase F — Tests + documentation (FINAL)

**Status (R1): Implemented for current surface; ongoing as operator coverage expands**

Artifacts:

- Tests: `tests/python/test_properties_*.py`
- Canonical storage/caching docs: `documentation/guides/Storage and Memory.md`

Deliverables:

- Tests:
  - compatibility checks (invalid combinations reject deterministically),
  - propagation checks (transpose/conjugate/scalar multiply),
  - persistence round-trip preserves properties.
- Docs:
  - user-facing docs for `properties` (semantics: gospel, not validated),
  - operator docs where properties change behavior.
  - a dedicated “Power users” section explaining that properties can override data truth and force structured algorithms.

Additionally:

- The “Adding Operations” protocol must be updated to include a properties checklist (how an op reads properties, whether it propagates/changes properties, and where that logic lives so it is not scattered).

- The caching model (derived caches) must be documented: how caches are validated, when they are cleared, and which metadata-only transforms preserve/transform which caches.

## Open questions (must be resolved before Phase B)

- Do we want to keep `is_unitary` propagation rules as written (transpose/conjugate clear; adjoint preserves), or should `is_unitary` be preserved more aggressively?
- Should `has_unit_diagonal` and `has_zero_diagonal` remain as stored boolean properties, or should `diagonal_value` be the only canonical representation (with shorthands computed on-demand)?
