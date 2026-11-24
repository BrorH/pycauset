# Feature: Value Bounds (known value range) and bounds-aware dtype selection

**Requested by:** maintainer (creative director), 2026-11-24.
**Status:** designed, not started. Scheduled in PLAN.md as Phase F1 (after the
operator consolidation in Phase 4.3, because it plugs into the single C++ result
type path).

## 1. The idea in one paragraph
Every matrix and vector already carries metadata (structure, dtype, shape, view
flags). We add one more piece: a **known value range**, a guaranteed interval
`[lo, hi]` that contains every element. Operations compute the result's range from
the inputs' ranges, *without touching the data* ("interval arithmetic"). The engine
then picks the **smallest dtype that is guaranteed to hold the result**, instead of
promoting blindly. Two int8 matrices whose values are known to be ≤ 2 add to an int8
result; a causal matrix (bits, range [0, 1]) squared gives path counts in [0, n], so
for n = 3,000 the result fits int16 instead of int32 (half the memory and disk).

**Name.** "Largest element" undersells it: we need the minimum too (signed types,
subtraction), and for floats a dtype is as much about *resolution* as size. The
concept is called **value bounds** (`ValueBounds` in C++, `A.value_bounds` →
`(lo, hi)` or `None` in Python). Precision/resolution is a separate axis that bounds
do *not* decide (§6).

## 2. Rules that make it trustworthy
1. **Sound, never optimistic.** The stored interval must always contain every
   actual element. It may be loose (wider than the truth), never too narrow. If we
   can't guarantee it, the bounds become **unknown** and the dtype's own full range is
   assumed (today's behaviour). A wrong bound means silent overflow, which is
   exactly what this feature exists to prevent.
2. **Cheap.** Bounds come from data only when we already touch every element
   (NumPy import, `fill`, generators like sprinkling, kernels that write the
   result). Otherwise they're derived from input bounds by arithmetic, which is O(1).
3. **One implementation, in C++.** `ValueBounds` lives on `PersistentObject` next to
   `scalar_` and friends. Propagation rules live next to `PromotionResolver`. Python
   only reads it.
4. **Persisted.** Saved in the `.pycauset` container metadata and restored on load.

## 3. Where bounds come from (sources)
| Source | Bounds |
|---|---|
| bit / bool matrices, causal matrices | [0, 1] by construction |
| `zeros`, `ones`, `identity`, `full(v)` | exact |
| NumPy import (`pc.matrix(ndarray)`) | one min/max pass during the copy (copy is already O(n²)) |
| Kernels writing a full result (matmul, elementwise) | derived from inputs (§4); optionally tightened by a min/max reduction fused into the write, if it's free |
| `A[i, j] = v` | widen to include `v` (O(1)) |
| Writes we can't see (NumPy buffer views, `write_block`, raw pointers) | set to **unknown** |
| Loading an old/foreign file without bounds | unknown |
| Explicit `pc.recompute_bounds(A)` (one pass) | exact; for when the user wants tight bounds back |

## 4. Propagation rules (interval arithmetic)
For inputs `A ∈ [a_lo, a_hi]`, `B ∈ [b_lo, b_hi]`:
- `A + B` → `[a_lo + b_lo, a_hi + b_hi]`; `A - B` → `[a_lo - b_hi, a_hi - b_lo]`
- elementwise `A * B` → min/max of the four corner products
- `s * A` (scalar) → scale and swap if `s < 0`; this also fixes the lazy-`scalar_`
  view story (F-001): bounds of a view = bounds of the base × scalar
- `-A` → `[-a_hi, -a_lo]`; `A.T`, slices, views → same bounds as base (slices may
  keep the base's bounds; that's sound, maybe loose)
- `A @ B` (inner dimension k) → `k ×` the corner-product range. For a causal matrix,
  `C @ C ∈ [0, n]`. Structure can tighten this: triangular inner dims are shorter.
- reductions (`sum`) → `count × range`
- division, `invert`, `solve`, eig, etc. → result is float; bounds unknown unless a
  rule is proven (no guessing)
- a property flag can tighten bounds (e.g. a permutation matrix is in [0, 1])

## 5. Using bounds to choose the result dtype
`PromotionResolver` today maps (dtype_a, dtype_b, op) → result dtype. New rule for
integer results:
1. Compute the result bounds (§4).
2. Pick the **smallest integer dtype in the same family** (signed if `lo < 0`) that
   contains the bounds, but never smaller than the narrowest input dtype (no
   surprise shrinking; see §7 for opt-in shrinking).
3. If even int64 can't be proven to hold the bounds: keep int64 and emit a
   `PyCausetOverflowRiskWarning` that states the proven bound (replaces today's
   heuristic sampled preflight in `bind_matrix.cpp:231-305`).

Policy change this implies: **integer arithmetic stops wrapping silently.** Today
int32 `2147483647 + 1` wraps to `-2147483648` (NumPy behaviour, and relies on UB,
F-006). With bounds, the engine widens when overflow is possible and stays narrow
when it's provably safe. For physics counts this is the right default.
**Divergence from NumPy:** NumPy picks result dtypes from input dtypes only
(NumPy 2, NEP 50); we also use value bounds. This must be stated plainly in the
docs' NumPy-compatibility page.

## 6. Floats and precision
Bounds still help floats with **range** (float16 overflows above 65,504, float32
above ~3.4e38): an op whose bounds exceed the result float's range widens or warns.
Bounds say nothing about **resolution** (how many significant digits you need), so
float precision choices stay governed by the existing promotion and precision rules.
For complex numbers, track bounds on |z| only.

## 7. Open design questions (technical lead decides during implementation)
- Bit/bool `A + B` (decided 2026-11-24): integer **counting**, not logical OR. With
  bounds the result of bit + bit is [0, 2] → int8 (today int32).
- Should results ever *shrink* below the input dtype (e.g. int64 inputs with tiny
  bounds → int8)? Default no; maybe a `pc.compact(A)` explicit call.
- Cost of fused min/max in kernels: measure; only keep it where free.
- Interaction with block matrices and out-of-core tiles: bounds per tile, merged.

## 8. Definition of done
- `ValueBounds` on every object; persisted; exposed read-only as `A.value_bounds`.
- Propagation for every op in the conformance spec table (TESTS.md), with
  "unknown" as the explicit fallback.
- **Soundness invariant test** in `tests/engine/test_invariants.py`: for every
  generated case, the actual min/max of the result lies inside the reported bounds.
- Dtype selection uses bounds for integer results; the heuristic preflight is deleted.
- Benchmarks: causal `C @ C` memory before and after; no slowdown > 2% on parity ops.
- One concise docs page: what bounds are, how they choose dtypes, how this differs from NumPy.
