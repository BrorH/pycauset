# Derived-value cache (maintainer request, session 8)

> "If you've got a giga matrix you've spent hours calculating the determinant of, it would be
> awesome if it was just stored alongside it ... BUT the second an object is changed ... it
> must be discarded. Should survive through lazy scalings though."

Replaces the removed Python cache (F-081: invalidated on `.set()` only, so `M[0,0] = 100`
left `sum` and `det` stale; propagated values through `+` without computing them). The
rule in CLAUDE.md becomes: no result caching outside this mechanism.

## Invariants
1. A cached answer equals what recomputation would return now (up to the rounding of the
   view rule in 4, which is documented). Never an answer for different data.
2. Invalidate before mutate: storage is marked changed before any byte can change.
3. Properties, sampling and guesses never fill the cache; only an engine computation does.

## Design
- **Owner:** the shared storage (the mapper that every view of the same payload holds), not
  the view object. It carries a `uint64 version` and a `DerivedCache` map
  `(quantity, params) -> value`.
- **Invalidation:** every path that can change bytes bumps `version` and clears the map:
  typed `set`, `fill`, `set_identity`, setitem, NumPy import into an existing object,
  `_debug_fill`, and every non-const access to the storage (`data()`, the mapper's raw
  pointer). Conservative on purpose: a non-const `data()` call invalidates even if nothing
  is written. A write through a view (`m.T[0, 1] = x`) changes the shared storage, so every
  view of it loses the cache.
- **Keyed on stored values:** the cache holds the quantity of the stored matrix S (no view).
  A view A = s · conj?(S) · transpose? derives its answer from the entry:

  | quantity | view rule | exactness |
  |---|---|---|
  | slogdet / det | logabs + n log abs(s); sign · (s/abs(s))^n, conjugated if conj | rounding of one multiply |
  | trace, sum | s · value (conj) | rounding of one multiply |
  | eigvals, eigvalsh | s · λ (conj); eigvalsh: sorted, sign flip reverses order | per-element multiply |
  | svdvals, norm 2 / fro / nuc / 1 / inf | abs(s) · value (1 and inf swap under transpose) | per-element multiply |
  | matrix_rank, cond | unchanged for s ≠ 0 | exact |

  Integer views (exact factor k) use the same rules in the result dtype. A window view
  (`m[1:4, 2:6]`) is a different matrix: no cache. Lazy closures (`A + A.T`) get their own
  entry keyed on the operands' versions (later step).
- **Persistence:** the container's metadata block (typed map, A/B header slots for atomic
  rewrites) gains a `derived` map saved with `payload_version`. A loaded file is memory-mapped
  and written in place, so the first mutation after load rewrites the metadata without the
  `derived` map (flip to the other header slot) *before* the payload changes. A crash between
  the two leaves no cache, never a stale one.
- **Large results** (inverse, eigenvectors, SVD factors, Cholesky / QR / LU factors): stored as
  extra payload sections of the same container (a format version bump), step D.4.

## Known limits
- A process that modifies the payload file without PyCauset (another program mapping it)
  is not detected; a cheap per-block checksum could close this later. Documented.
- Derived view answers are one multiply away from recomputation (floating rounding);
  powers of two are exact. If bit-identical results are required, the view rule can be
  restricted to exact factors.

## Steps (PLAN phase D)
- D.1 C++ storage version counter + invalidation on every mutable path; a test that walks
  every binding that writes and asserts the version moved (the atlas can enumerate them).
- D.2 Cache map on storage; entries for slogdet/det, trace, sum, norms, rank, eigvals,
  eigvalsh, svdvals; the view rules above in one place (next to ViewValue.hpp).
- D.3 Persistence: `derived` metadata + invalidate-before-mutate on mapped files.
- D.4 Matrix-valued results as container sections.
- D.5 Closures and expression nodes (with phase L).
- Gates: the atlas runs every cached op twice (cold, warm) and after a mutation, and checks
  all three against NumPy; a crash-injection test for D.3.
