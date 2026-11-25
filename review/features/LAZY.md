# Feature: Lazy evaluation (views and deferred expressions)

**Requested by:** maintainer (creative director), 2026-11-24: "make sure this is implemented
for ALL dtypes and matrix types … very thoroughly implemented, documented and thought through."
**Status:** L1 in progress (session 5): views read at kernel speed (F-043), integer scale views for dense/triangular (F-042), free Diagonal transpose (F-044); remaining L1 items listed under F-042. Scheduled in PLAN.md as Phase L (L0–L6). Findings: F-042
(scalar views missing for integer/bit), F-043 (consuming a view costs 2×), F-044 (free
transposes missed).
**Probe:** `review/probes/view_semantics_probe.py` (which class × op is a view today, value
check against NumPy, mutation-leak check).

---

## 1. What "lazy" means here, in one paragraph
An operation is *lazy* when it returns a result **without computing its elements**. It records
*how* to compute them, and the work happens later, ideally fused into some other pass that
had to run anyway, or never at all. `2 * A` can return "A's storage, times 2" in O(1). The
factor is applied when someone reads the elements, at no extra cost because that pass reads
them anyway. The approach scales well: an n = 100,000 matrix is 80 GB of float64, so every
avoided materialization is 80 GB not written to disk.

## 2. The one rule
**Laziness is unobservable except through speed and memory.** A lazy result has the same
values, dtype, structure, Python class, shape, error behaviour and persistence format as
computing it immediately ("eager"). A user can't tell, and never needs to tell, whether
something is lazy.

Consequences:
- **No second-class objects.** The old `LazyMatrix` returned an object you couldn't sum,
  transpose or save. That's why it was removed. A lazy result is an ordinary `FloatMatrix`
  (or whatever eager would produce) whose elements happen to be defined by a recipe.
- **Dtype and structure are decided eagerly,** by the same resolver as eager
  (`PromotionResolver`, `StructureResolver`). Only the element values are deferred.
- **Errors are raised at creation** whenever eager would raise them (shape mismatch,
  unsupported dtype, singular-by-structure). Laziness never moves an error to a later,
  unrelated line.
- **Value semantics, no aliasing.** `B = 2 * A; A[0, 0] = 9` leaves `B` unchanged. The probe
  confirms copy-on-write already guarantees this for every class and every view op today
  (0 leaks). NumPy differs for `A.T` (shared memory); we deliberately keep value semantics
  and document the difference.
- **Whether to stay lazy or materialize is a performance decision,** so heuristics are
  allowed to make it (project rule: guesses may inform, never decide). This is only true
  because of the exactness contract below.

## 3. Exactness contract (when may a lazy result differ in rounding?)
Two tiers. Every lazy rewrite in this document is labelled E or A.

- **Tier E, bit-identical to eager.** Same element values to the last bit. This holds:
  - for **integer and bit dtypes always**. Integer arithmetic here wraps (mod 2^k, NumPy
    semantics), which is a ring, so any algebraic regrouping is exact: `(x·a)·b ≡ x·(a·b)`,
    `(A+B)x ≡ Ax + Bx`, overflow included;
  - for index maps (transpose, conjugate, slicing) and negation, which do no rounding;
  - for float scaling when the folded factor is a power of two, or when there's only one
    factor;
  - for any rewrite that applies **the same operations in the same order**, just at a
    different time (a per-element recipe evaluated on read).
- **Tier A, algebraically equal, deterministic, may differ in the last bits.** Floats only.
  Examples: folding `2.1 * (3.3 * A)` into `6.93 * A` (one rounding instead of two, so
  usually *more* accurate), `(A+B)x` evaluated as `Ax + Bx`, `A@B@v` reordered as `A@(B@v)`.
  Rules:
  1. A tier-A plan may depend only on the **expression** (ops, shapes, dtypes, structures,
     factor values), **never on machine state** (free RAM, timing, thread count). The same
     program gives the same bits on every machine.
  2. **Guards against changing inf/nan/overflow behaviour:** a folded float factor must be
     finite, and nonzero unless an input factor was zero. Otherwise don't fold; keep the
     factors as a chain (tier E). Example: `1e200 * (1e200 * A)` with `A = 1e-300` is
     `1e100` eager but `inf` if folded.
  3. Documented in the user docs as: "floating-point results may differ from step-by-step
     evaluation in the last digits; they're deterministic."
- **Default:** tier A is on for floats (it's how BLAS, NumPy's `einsum(optimize=True)` and
  every compiler at `-O2` already behave, and float matmul is already not bit-identical to
  NumPy). **Maintainer veto point:** if bit-identical step-by-step float semantics matter
  more than speed, tier A is switched off and float folding stops at tier E. Integer and
  bit laziness is unaffected either way.
- The current out-of-core tiler already changes summation order with tile size (which
  depends on RAM). That's a pre-existing tier-A-on-machine-state violation; it's recorded
  under F-040 follow-ups and fixed by making tile sizes a deterministic function of shape.

## 4. Vocabulary
- **Storage:** the bytes (RAM or memory-mapped file) of one dtype and structure.
- **View:** storage + an *index map* + a *value map* + a *logical dtype/structure*. O(1) to
  create. Reading an element costs O(1).
- **Index map:** transpose, conjugate-transpose, row/column offsets and lengths (slicing).
- **Value map:** a short per-element recipe applied on read: scale by a factor, negate,
  conjugate, cast to the logical dtype, add a constant on the diagonal (shift).
- **Expression node (lazy node):** a result defined by other matrices: sum, product, power,
  inverse, closure (`A ± Aᵀ`). Reading one element can cost more than O(1) (O(n) for a
  product element).
- **Consumer:** an operation that reads its operands: element access, export, save, kernels
  (element-wise, matmul, matvec), reductions, eigensolvers.
- **Materialize:** evaluate a lazy result into fresh storage. Always allowed (tier E/A
  per §3); the result is the same matrix, just no longer lazy.

## 5. What exists today (probe, 2026-11-24, 6×6, 14 dense dtypes + 7 structures)
| | `k*A`, `-A`, `A/k` | `A.T` | `A.conj()`, `A.H` |
|---|---|---|---|
| float16/32/64, complex64/128 dense | view ✔ | view ✔ | view ✔ |
| int8–64, uint8–64 dense | **copy** (F-042) | view ✔ | view ✔ |
| bit dense, TriangularBit (causal) | **copy** (F-042) | view ✔ | view ✔ |
| TriangularFloat, AntiSymmetric | view ✔ | view ✔ | view ✔ |
| TriangularInteger | **copy** (F-042) | view ✔ | view ✔ |
| Diagonal, Symmetric | view ✔ | **copy** (F-044: they're their own transpose) | view ✔ |
| Identity | view ✔ | view ✔ | view ✔ |
| every vector dtype | **copy** (F-042) | view ✔ | (n/a) |

All values correct; no mutation leaks. Cost of consuming views today (3000², 4 cores):
`A+B` 32 ms, `2A+3B` **62 ms** (F-043: the scaled operands drop to a slow per-element path;
a view should be free on read, as it is in BLAS with `alpha`).

Existing machinery to build on: `PersistentObject` already carries `scalar_`
(`complex<double>`), `is_transposed_`, `is_conjugated_`; persistence already stores the
scalar; copy-on-write already gives value semantics.

## 6. Catalogue: every lazy form worth having, ranked by payoff
"Win" is what it saves; "class" is asymptotic (changes big-O), memory (changes what fits),
or constant (bandwidth factor).

### L1 Value and index views for every dtype, structure and vector (prerequisite for all else)
- **What:** `k*A`, `A*k`, `A/k`, `-A`, `+A`, `A.T`, `A.H`, `A.conj()`, and lazy **casts**
  (logical dtype ≠ storage dtype): `2 * C` on a bit causal matrix is an int view over the
  bit storage (1/32 the memory of int32, 1/64 of float64). Every one of the 16 dtypes, every
  structure, matrices and vectors.
- **Factor storage:** exact and typed. An `int64` factor for integer/bit logical dtypes
  (applied with wrapping arithmetic; tier E, overflow identical to NumPy), and
  `complex<double>` for float/complex (tier E for a single factor; tier A when folding).
- **Value map is a short recipe, not one number,** so the cases that aren't a single factor
  are still lazy and exact. For example `int32 * 3.5` under our promotion rules
  (stays int32, truncated per element) is "cast → scale → truncate" applied on read. Recipe
  length is capped (say 4 steps); past the cap the result is materialized, which is a pure
  performance rule.
- **Free transposes:** `Diagonal.T`, `Symmetric.T`, `Identity.T` return the same storage, no
  flag (F-044). `AntiSymmetric.T` = factor −1. `Hermitian` when it exists: `.H` is free.
- **Readers must honour the view (the F-001 lesson).** Every consumer reads through one
  central function per tile, `ViewSpec::apply(tile)` (§8), at kernel speed (F-043): the
  factor goes into BLAS `alpha`, into the element-wise kernel's inner loop, into the
  streaming tiler (closes the F-040 follow-up), into reductions as a final multiply
  (`sum(kA) = k·sum(A)`, tier E for one factor).
- **Win:** O(n²) → O(1) per scale, sign or cast; chains fold. Disk: no new file per op.
- **Exactness:** E (ints, bits, index maps, single factors); A only for folding float factors.

### L2 Structural closures and shifts (big for causal sets)
- **What:** `C + C.T` (symmetric) and `C − C.T` (antisymmetric) as views over **one
  triangular storage**. An element read picks `C[i,j]` or `±C[j,i]`. Also the shift
  `A + c·I` (and `A − λI`) as a diagonal-shift entry in the value map.
- **Why it matters:** the comparability matrix and the Pauli–Jordan operator
  (`i·a·(C − Cᵀ)`) are the central R1 objects. For n = 100,000, the causal matrix is 0.6 GB
  as bits; `C − Cᵀ` materialized as float64 is 80 GB. As a closure view it's still 0.6 GB.
  The StructureResolver already knows `C + Cᵀ` is symmetric (F-039), so the closure view
  plugs into the existing resolver; the eigensolver then picks Lanczos automatically.
- **Consumers:** matvec computes `Cx ± Cᵀx` with the bit-triangular kernels (two half-cost
  passes, no dense matrix), so `eigvals_skew` and `eigvals_arnoldi` on Pauli–Jordan run
  without ever materializing it. Element access O(1). Shift: matvec adds `c·x`; eigenvalues
  shift by `c` (L4).
- **Win:** memory class (64–128× less storage), which is the difference between "fits" and
  "doesn't".
- **Exactness:** E for element reads; matvec `Cx − Cᵀx` vs materialized `(C−Cᵀ)x` is E for
  ints and bits, A for floats.

#### L2 implementation plan (session 5)
- **L2a, eager but packed (small, exact) — DONE session 5** (n = 6000 causal: `C − Cᵀ` 72 MB
  packed vs 144 MB dense; tests in `test_regressions.py`): the structure resolver recognises `A ± B` where B is
  the transpose view of A (same mapper, opposite transpose flag, equal factor and conjugation,
  no offsets). `A + Aᵀ` is exactly symmetric (floating-point addition commutes) and `A − Aᵀ`
  exactly antisymmetric, so the result is stored packed: half the memory for every dtype,
  and `eigvals_skew` / Lanczos apply directly. For a causal matrix, `C − Cᵀ` becomes an
  int32 AntiSymmetricMatrix (n²/2 × 4 bytes instead of n² × 4); with value bounds (F1) the
  dtype drops to int8.
- **L2b, the closure view (the 32–64× saving) — DONE session 5** (`BitClosureMatrix`;
  n = 4000: no extra storage vs 32 MB packed, build 0.9 ms vs 49 ms, same Arnoldi spectrum;
  `tests/engine/test_closures.py` compares every consumer with the eager oracle
  `_set_lazy(False)`; `A.is_lazy`, `pc.materialize(A)` added). Word-level closure matvec done
  (set-bit gather + column-block scatter): n = 4000 Arnoldi 1.47 s -> 0.20 s; n = 20,000 top-6
  spectrum of C − Cᵀ in 1.8 s at 99 MB peak RSS. Follow-ups: dense `eigvals_skew` is O(n³) whatever the storage
  (L7.1 real skew solver). **Session 5:** `eigvals_skew` now uses the Hermitian solver on iA
  (zheevd, exact ± pairs, deterministic order): n = 1500 0.65 s -> 0.36 s. `eig_skew` (vectors)
  still uses dgeev. Original requirements: a `ClosureMatrix` class (C++) deriving the
  Symmetric/AntiSymmetric structure bases, holding a shared pointer to the source triangular
  storage plus a sign and a logical dtype. Requirements found while designing it:
  1. `visit_matrix` (DTypeDispatch.hpp) assumes the storage class matches (structure, dtype);
     it needs a branch for closure objects (dispatch on the *source* storage type) or every
     typed consumer breaks with "object does not match its tag".
  2. Element reads: `C[i,j] ± C[j,i]` through the source's typed get, then cast to the logical
     dtype (the bit -> int32 promotion).
  3. Fast consumers: matvec = `Cx ± Cᵀx` with the bit-triangular kernels (both exist:
     `cpu.matvec.tri_bit_x_*`), so Arnoldi/`eigvals_skew` run without materializing.
  4. Everything else materializes once (packed, L2a) and caches the result.
  5. Persistence saves the source storage plus the closure flag (no n² payload).
  Gate: the reader × view grid extended with closure objects, and an "allocation" test that
  `eigvals_skew(C − C.T)` never allocates an n² buffer (MemoryGovernor counters).

### L3 Operator expressions: sums, products, powers, inverses (asymptotic wins)
- **What:** lazy nodes `αA + βB (+ …)`, `A @ B`, `A ** k` (matrix power), `inv(A)`, built on
  L1 views as leaves.
- **Consumers that exploit them:**
  - **Matvec and the iterative solvers built on it** (Arnoldi/Lanczos, power iteration, CG):
    `(αA+βB)x = αAx + βBx`, `(A@B)x = A(Bx)`, `Aᵏx` = k matvecs, `inv(A)x` = solve.
    Never forms an n² intermediate.
  - **Product reordering:** a chain `A @ B @ C @ v` is evaluated in the cheapest order
    (classic matrix-chain ordering on shapes). Measured: `(A@B)@v` 236 ms vs `A@(B@v)`
    41 ms at n = 3000. The gap grows like n (100× at n = 300,000).
  - **Partial reads of products:** `trace(A@B) = Σ A∘Bᵀ` (O(n²) not O(n³)); `sum(A@B) =
    (1ᵀA)(B1)` (two matvecs); `diag(A@B)` O(n²); a single element `(A@B)[i,j]` O(n).
  - **Inverse as a solve:** `inv(A) @ B` becomes `solve(A, B)` (about 3× cheaper and more
    accurate), `inv(inv(A)) = A`, `inv(A.T) = inv(A).T`, `inv(kA) = inv(A)/k`; triangular
    → triangular solve. Shift-invert eigenvalue problems (`inv(A − σI)`) come for free
    from L2 + L3.
  - **Powers:** `matrix_power(C, k) @ v` (path counting on causal sets) is k bit-matvecs
    instead of log k dense n³ products.
- **When a node is read element by element or exported,** it's materialized once and
  cached (same object, now concrete). Materializing a product costs what eager would have
  cost, never more.
- **Exactness:** E for integer/bit (ring); A for floats (reordering changes rounding).
- **This is where your `a*(A+B)` lands:** as an element-wise result it's already optimal
  (one add pass; `a*` is a free L1 view, measured 33 ms vs 32 ms for `A+B`). The payoff is
  when `a*(A+B)` is consumed by a matvec, eigensolver, trace or sum: then `A+B` is never
  formed at all.

### L4 Property algebra (answers without touching data)
- **What:** results that follow from cached properties of the operands through views and
  nodes: `sum(kA) = k·sum(A)`, `trace(Aᵀ) = trace(A)`, `trace(A+B) = trace A + trace B`,
  `norm_F(kA) = |k|·norm_F(A)`, `det(kA) = kⁿ·det(A)`, `eigvals(kA) = k·eigvals(A)`,
  `eigvals(Aᵀ) = eigvals(A)`, `eigvals(A + cI) = eigvals(A) + c`, value bounds of `kA`
  (Phase F1, same mechanism).
- **Ties to:** the decision that properties are owned by C++ (RESTRUCTURE §3.1 item 2):
  one property cache, invalidated on write, propagated through views.
- **Win:** O(n²)–O(n³) → O(1) when a property is already known. Eigenvalue reuse alone can
  save hours on large Pauli–Jordan studies (scaling conventions differ between papers).
- **Exactness:** E for exact properties of ints, bits and index maps; A for float rescaling
  (`k·λ` vs eigvals of the scaled matrix).

### L5 Element-wise fusion (`A + B + C`, `a*A + b*B − C`): constant factor, mostly for disk
- **What:** a tree of element-wise ops evaluated in one tiled pass, with no temporaries.
- **Honest payoff:** in RAM it's bounded by memory traffic: eager `A+B+C` moves 6n²
  elements, fused moves 4n², so at most about 1.5× (NumPy measures 1.46×). Out of core the
  same ratio applies to disk I/O, and temporaries no longer need disk space (a 3-operand
  expression at n = 100,000 avoids an 80 GB temporary file). That's the real case for it.
- **Why the old expression engine showed no gain:** our element-wise kernel is about 2×
  slower than NumPy (F-012) and views cost 2× more to read (F-043). Overhead swamped the
  bandwidth saving. Fusion only pays after F-012/F-043 are fixed.
- **Scheduling:** after L1–L3, and only if a benchmark after F-012 shows ≥ 1.3× in RAM or a
  meaningful disk saving out of core. Built on L3's sum nodes (a sum node consumed by an
  element-wise kernel is fusion), so no second expression system.
- **Exactness:** E (same ops, same order per element).

### L6 Constant matrices and slices (cheap, low priority)
- `zeros/ones/full(n, n)`, `IdentityMatrix` as storage-free constants: O(1) memory;
  `ones @ x` is a broadcast sum. Today `zeros` allocates (sparse files make it cheap on disk
  anyway, so this is minor).
- Slices `A[i:j, k:l]` as index-map views (today they copy; to verify). Uses the existing
  offset fields.
- **Exactness:** E.

### L7 Causal-set-specific operators (maintainer request: "we know what the math will look like")
The physics layer (`field.py`) computes K_R = aC(I − baC)⁻¹, iΔ = i(K − Kᵀ) and the SJ Wightman
function W in dense complex NumPy, outside the engine. These are the forms worth building in.
Each was checked on a real causal matrix (n = 600, `review/LOG.md` session 5).

1. **Pauli–Jordan in real arithmetic.** K is strictly upper triangular (C is nilpotent), so
   iΔ = i·(K − Kᵀ) is fully described by one real triangular K plus "antisymmetric closure × i".
   Storage: 4n² bytes (float64 triangle) instead of 16n² (complex128 dense), so 4× less. Its
   spectrum is ±λ pairs of the real antisymmetric A = K − Kᵀ: a real skew solver
   (`eigvals_skew` exists) instead of complex Hermitian `eigh` (measured 22 ms real vs 87 ms
   complex at n = 600), and only half the pairs are needed. Builds on L1 (complex factor `i`
   on real storage) + L2 (closure).
2. **SJ Wightman function without complex storage.** Exact identity (verified):
   W = ½(|A| + iA) with |A| = √(AᵀA). The imaginary part is iΔ/2, already stored (item 1),
   so W is a lazy sum of one real **symmetric** matrix |A| (packed: 4n² bytes) and a view of
   K: 4× less memory than complex dense W, and |A| comes from the positive half of the skew
   decomposition, computed in real arithmetic (use the skew decomposition, not √(AᵀA): squaring
   loses digits). Restrictions to a region (entanglement entropy) are slice views (L6).
3. **Matrix-free propagator.** K x = aC·(I − baC)⁻¹x = one triangular solve against the
   **bit** matrix plus one bit matvec; K is never formed. Memory at n = 100,000: C as bits
   0.6 GB versus K as a float64 triangle 40 GB. Top-k Pauli–Jordan / truncated SJ modes (a
   standard truncation in the literature) then run by Arnoldi at 10–100× larger n. Needs a
   bit-matrix triangular-solve kernel (y_i = x_i + β Σ_{j>i} C_ij y_j).
4. **Interval statistics without forming C².** Interval sizes are (C²)_xy. The BD action and
   d'Alembertian, link counts and Myrheim–Meyer dimension use them only through reductions:
   histograms over relations, masks, row sums, or applying the BD operator to a field. A fused
   "product + reduction" node computes C² tile by tile and reduces immediately: O(tile)
   memory instead of 4n² bytes (int32 C² at n = 100,000 is 40 GB). Special case already
   verified: Σ(C²) = (1ᵀC)(C1), two bit matvecs, O(n²) instead of O(n³). Exact (integers).
5. **Causal matrix generated from coordinates** (sprinkled causets, lower priority): store the
   n × d coordinates (16 MB at n = 10⁶, d = 2) instead of C (62 GB of bits) and recompute
   relations inside matvecs with the same predicate (bit-identical). Trades compute for I/O;
   worth it only beyond RAM. Measure before building.

### Rejected or not worth it
- **Vector and diagonal scale views** (session 5 decision): these objects are O(n); a scaling copy
  costs ~1 ms per million elements, while each view type adds reader surface that must be
  proven correct. Views stay for the O(n²) structures. Revisit only if profiling says otherwise.
- **Lazy element-wise functions in general** (`exp(A)`, `A ** 0.5` element-wise): a view
  would recompute the function on every read. Only kept as fusion inside L5.
- **Lazy comparisons and masks** (`C & ~(C @ C)`) as standalone objects: eager. Their common
  uses (link counts, BD operator) are covered by the fused reductions in L7.4.
- **Laziness for small matrices** (below ~10⁴ elements): bookkeeping costs more than it saves;
  the planner materializes them.
- **Dtype-deferred allocation** (`pc.zeros` without dtype deciding its type later, F-041b):
  that's guessing a dtype, not laziness. Separate question, already with the maintainer.

## 7. Lifetimes, memory, persistence
- **Lifetimes:** views and nodes hold their operands' storage alive (shared ownership, as
  today). A node never holds an operand whose storage is later mutated: copy-on-write
  guarantees a snapshot.
- **Memory cap on deferred work:** a node tree whose operands are kept alive *only* by it
  pins disk and RAM. Rule: depth ≤ 8 and pinned-but-otherwise-dead bytes ≤ a fraction of
  the memory budget, else materialize. Both are performance rules, legal under §2.
- **Save/load:**
  - L1 views save as storage plus the view spec (today the format stores the scalar; this
    extends it to the typed factor, recipe and logical dtype). Loading gives the same view,
    O(1).
  - L2 closures save as triangular storage plus the closure flag.
  - L3 nodes **materialize on save.** A file must be self-contained: it can't depend on
    other files that might be deleted, and the maintainer wants full control of the
    `.pycauset` directory.
  - The file format records the exactness tier of its values, for provenance.
- **Spill directory:** unchanged (`.pycauset` in the working directory).

## 8. Architecture (C++ owns all of it; Python is a thin wrapper)
- **`ViewSpec`** (replaces the loose `scalar_`/`is_transposed_`/`is_conjugated_` fields):
  - index map (transpose, conjugate, offsets, lengths);
  - value recipe (≤ 4 steps of: typed scale, negate, conjugate, cast, truncate, diagonal
    shift, closure sign);
  - logical dtype and structure.

  It provides `apply(tile_in, tile_out)`, vectorised per dtype pair. This is **the only
  place** view semantics live. Every kernel either understands a ViewSpec natively (BLAS
  alpha, transposition flags) or calls `apply` on its tiles. A unit test asserts each
  dtype × structure × recipe step reads identically through every consumer.
- **`ExprNode`** for L3: an immutable DAG (sum, product, power, inverse) whose leaves are
  views. It implements the matrix interface. `matvec`, `trace`, `sum`, `diag`, `element`
  have node-aware implementations; anything else calls `materialize()`, which caches the
  result.
- **Planner:** product-chain ordering and the materialize-or-not decisions. It's a pure
  function of the expression (§3 rule 1), with one debug switch, `_set_lazy(False)`, that
  forces eager everywhere (the test oracle).
- **Generated support table:** the dtype × structure × class table from Phase 4.6 (F-032)
  also generates the ViewSpec dispatch, so "every dtype and every structure" is
  enforced by construction, not by memory.
- **Python surface:** none beyond the existing classes. Introspection for power users:
  `A.is_lazy` (bool), `pc.materialize(A)` (forces evaluation, returns `A`), and the kernel
  trace for debugging.

## 9. Interactions
- **Value bounds (F1):** bounds propagate through views and nodes for free (`bounds(kA) =
  k·bounds(A)`, `bounds(C + Cᵀ) ⊆ [0, 2]`), and lazy casts make bounds-chosen small dtypes
  cheap. Implement after L1 so both share the ViewSpec.
- **Out-of-core (StreamingManager):** the tiler reads through `ViewSpec::apply`, which
  removes the F-040 "never stream views" restriction; sum nodes stream as fused tiles (L5).
- **Properties cache:** L4 is the cache's propagation rule set.
- **Promotion and structure resolvers:** unchanged; called eagerly at node creation.
- **GPU:** on hold; ViewSpec would map onto the CUDA kernels the same way.

## 10. Test strategy
1. **Oracle: lazy ≡ eager.** Every test in the conformance grid runs twice, normal and with
   `_set_lazy(False)`. Tier E ops are compared bit for bit; tier A ops within a stated ulp
   bound. One switch turns the whole existing suite into a laziness test.
2. **Every reader × every view:** for each dtype × structure, build `k*A`, `-A`, `A.T`,
   `A.H`, cast views and closures, and read each one through every consumer (element get,
   export, sum, matmul, matvec, element-wise, streaming, save/load). This is the test whose
   absence hid F-001.
3. **Random expression trees:** a seeded generator builds random L1–L3 expressions and
   compares against NumPy and the eager oracle.
4. **Asymptotic tests, not timing tests:** assert *what work happened* via the kernel trace
   and operation counters. For example, `A@B@v` performs two matvecs and no matmul;
   `eigvals_skew(C − C.T)` never allocates an n² buffer. No flaky timings in CI.
5. **Value semantics:** mutation after view creation never leaks (the probe, made a test).
6. **Persistence round trip** of every view kind, and of nodes (which materialize).
7. **Benchmarks** (outside CI, recorded in BASELINE.md): `2A+3B` vs `A+B` (target: equal),
   `C−Cᵀ` eigensolve memory, `A@B@v`, fusion at n = 3000 in RAM and n = 30,000 on disk.

## 11. Phases (in PLAN.md as Phase L)
| phase | content | depends on | exit criterion |
|---|---|---|---|
| L0 | F-032 table + renames (4.6); single operator path (4.3); F-043 view-read speed | — | `2A+3B` ≈ `A+B`; generated class table |
| L1 | ViewSpec; typed exact factors; scale, negate, conj, transpose, cast views for all 16 dtypes × all structures × matrices and vectors; free transposes; streaming reads views | L0 | oracle + reader×view grid green; probe table all "view" |
| L2 | closures `A ± Aᵀ` over triangular storage; diagonal shift; solvers consume them | L1 | Pauli–Jordan eigensolve at n where dense doesn't fit |
| F1 | value bounds (shares ViewSpec) | L1 | per VALUE_BOUNDS.md |
| L3 | ExprNode: sums, products (chain ordering), powers, inverse-as-solve; matvec/trace/sum/diag consumers | L2 | asymptotic tests green |
| L4 | property propagation through views and nodes | L3, C++ properties (4.3) | — |
| L5 | element-wise fusion via sum nodes | L3, F-012 | benchmark ≥ 1.3× or disk saving shown |
| L6 | constants, slices | L1 | — |
| L7 | causal-set operators: real Pauli–Jordan / SJ (7.1–7.2), matrix-free K (7.3), fused interval statistics (7.4); 7.5 after measurement | L2, L3 | SJ memory 4× lower; top-k PJ at n where K doesn't fit |

## 12. Lessons from the removed expression engine (`d564c7f`)
1. It returned a second-class object (`LazyMatrix`). Rule 2 forbids that.
2. It covered only float dtypes. L1 is generated from the full dtype table.
3. No consumer exploited it (no matvec, reordering or trace algebra). It only deferred
   element-wise work, the smallest possible payoff (§L5).
4. It was measured slower than eager, because kernel overhead (F-012) swamped fusion. Fix
   kernels first, then fuse.
5. It lived half in Python and half in C++. This design is C++ only.

## 13. Open points for the maintainer (non-technical)
- **Tier A default (§3):** may float results differ from step-by-step evaluation in the last
  digits, deterministically, in exchange for the speedups? Tech-lead recommendation: yes
  (integers and bits are exact either way).
- **NumPy difference on `A.T`:** pycauset keeps value semantics (writing into `A` never
  changes an earlier `A.T`). Recommendation: keep it; document it.
