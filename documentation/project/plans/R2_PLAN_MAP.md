# R2 Feature Plan Map

**Status**: Implemented, the R2.0/R2.1/R2.2 physics is done (free scalar field core:
retarded/advanced propagators, Pauli–Jordan `iΔ`, Sorkin–Johnston Wightman); the R2E
engine/optimization phase is partially done (elementwise SIMD, SRP-2 catalog 3/4, eigen-cache
persistence, parity baseline) with the remainder tracked live in `R2_ROADMAP.md`. This document
is the authoritative R2 feature map.
**Audience**: Creative director + contributors.
**Companion**: `R2_API_DESIGN.md` holds the deeper rationale and tradeoffs. Where they
differ, **this document wins** (it records the director's locked decisions).

---

## 1. Principles (locked)

1. **Causal sets are the primary citizens.** The continuum spacetime and its coordinates are
   *provenance and attachments*, not the object.
2. **Professional: never guess.** PyCauset must never infer what the user wants from a class
   name or a heuristic. If a coefficient isn't known, we raise and ask for it explicitly -
   we never silently pick a "probably right" value. (The one exception: a *predefined library*
   spacetime may ship with documented coefficients, that is not guessing, it is authored.)
3. **Fun and easy.** A user should reach a working causal set in a few lines; advanced control
   (batch hooks, manual coefficients, exotic signatures) is opt-in, never required.
4. **Correct first.** Invariants are validated; a non-transitive matrix is never allowed to
   masquerade as a causal set.
5. **Arbitrary dimension and geometry.** The signature is a first-class property, not a hidden
   Lorentzian assumption.
6. **Docs and tests in-step, never deferred.** Every R2 step ships its own documentation and tests
   *as part of that step's Definition of Done*, there is no trailing "docs/test phase." Deferring
   them loses information to memory and lets bugs hide; so both are non-negotiable per step.

---

## 2. Feature map at a glance

| # | Area | Desired features | Priority |
| :- | :-- | :-- | :- |
| 3 | Signatures & dimension | First-class signature; arbitrary dimension; causal order only when Lorentzian | P0 |
| 4 | Spacetime library | Extensive built-in library: flat, curved/cosmological, synthetic | P0/P1 |
| 5 | Custom spacetime creation | Tier-0 builder, Tier-1 subclass, Tier-2 batch (Spacetime Studio = future bonus) | P0/P1 |
| 6 | Modify & compose spacetimes | Subclass or wrap a library spacetime; save/load the result | P0 |
| 7 | Fields & coefficients | Spacetime-owned `scalar_coeffs`; manual override; never guess (see `R2_FIELDS_PHYSICS.md`) | P0 |
| 8 | Coordinates & embedding | Hybrid: regenerate-from-seed by default, attach explicit embedding when asked | P0 |
| 9 | Validation | Eager partial-order validation with a `validate=False` escape hatch | P0 |
| 10 | Visualization | Plotly-only; hybrid call surface (methods + lazy top-level + `show`); subset + warning + bypass; higher-D deferred | P0/P2 |
| 11 | Persistence | Save/load spacetimes (built-in + custom + modified) by registry/recipe | P0 |
| 12 | Physics core | Propagators K_R/K_A, Pauli–Jordan iΔ, Wightman/Sorkin–Johnston, correlators, vevs, flawless; dimension estimators, entanglement entropy (see `R2_FIELDS_PHYSICS.md`) | P0/P1 |
| 13 | Engine & optimization | Folded-in post-R1 program: ≥0.90× NumPy parity, tiled CPU engine, GPU parity, streaming-everywhere, SRP-2 catalog, eigen-cache (R2E) | P0/P1 |

---

## 3. Signatures & dimension

**The design.** A `Spacetime` exposes its **signature** as `(t, s)` = (timelike, spacelike),
with `dimension() = t + s`. Index `0..t-1` are timelike; for Lorentzian `t = 1`, index 0 is
time (matching today's code). No other signature is assumed anywhere.

**Why this matters for QG.** A causal set needs a *causal order* `≺`. That order comes from a
Lorentzian metric (one timelike direction). So:

| Signature | `sample()` | `is_causal()` | Produces |
| :-- | :-: | :-: | :-- |
| **Lorentzian** `(1, d-1)` | yes | yes (timelike future) | a **causal set**, the standard path |
| **Euclidean** `(0, d)` | yes | undefined → raises | a **point process** (Euclidean QFT, correlations, embedding checks), no causet |
| **Multi-time** `(t>1, s)` | yes | only if user supplies a "future" convention | research-grade; field coefficients always manual |

**The honest constraint to document:** "arbitrary dimensions for R2" means *sprinkling,
geometry, and causality* work in any `d`. The automatic field coefficients `(a, b)` are known
only for specific cases (2D/4D Minkowski, and whatever the library ships with proofs/docs).
Everywhere else `scalar_coeffs()` raises `NotImplementedError` and the user supplies `a, b` by
hand. This is principle #2 applied consistently, **arbitrary dimension never means guessed
physics.**

**R2 deliverable:** generalize Minkowski Diamond/Cylinder/Box to arbitrary `d` with correct
volume, uniform sampling, and causality; add the signature property to the `Spacetime` ABC.

---

## 4. Spacetime library (extensive)

A curated, first-class (documented + tested) library, organized by category. Names below are
illustrative, not final.

| Category | Candidates | Priority |
| :-- | :-- | :- |
| **Flat Minkowski** | `MinkowskiDiamond`, `MinkowskiCylinder`, `MinkowskiBox`, generalized to arbitrary `d` | P0 |
| | `MinkowskiSpace` (with explicit cutoff/region), `MinkowskiCone` | P1 |
| **Curved / cosmological** | `DeSitter` (dS), `AntiDeSitter` (AdS) in `d` dims via embedding/hyperboloid | P1 |
| | `FLRW` (scale factor `a(t)`, open/flat/closed slices) | P1 |
| | `ConformallyFlat` (Minkowski + conformal factor) | P1 |
| **Black holes** | `Schwarzschild` (Eddington–Finkelstein / Kruskal region) | P2 / later |
| **Periodic / compact** | torus/cylinder generalizations in higher `d` | P1 |
| **Synthetic / test** | `Chain`, `Antichain`, `TransitivePercolation`, `RandomDAGOrder`, `KleitmanRothschild`, `IntervalOrder`, `Dimension2Poset`, `ProductOrder`, `Poset`, poset generators ("a causet is just a poset"; see library doc) | P2 |

Each library spacetime ships with: its **signature**, **volume**, **uniform sampler**, **causal
predicate**, **documented field coefficients where known**, and an optional **embedding/boundary**
for plotting. That is the whole contract, everything else is uniform machinery.

---

## 5. Custom spacetime creation, the "easy ladder"

Four rungs, each built *on top of* the one below (no magic, sugar over an explicit contract):

### Rung 0, Declarative builder (`spacetime.create`), the *even easier* path

For the common cases, no class at all. A recipe is an explicit configuration, never an inference:

```python
from pycauset import spacetime

st = spacetime.create(
    dimension=3,
    signature=(1, 2),          # Lorentzian 1+2
    domain="box",              # box | diamond | cylinder | ball | none
    metric="flat",             # flat | de_sitter | anti_de_sitter
    time_extent=4.0,
    space_extent=(2.0, 2.0),
)
```

`create()` assembles a configured `Spacetime` under the hood. Every parameter maps 1:1 to a
concrete setting; there is no hidden inference, so principle #2 is preserved.

### Rung 1, Subclass (the semi-easy path, from R2_API_DESIGN §2.1)

Three methods get you a working spacetime; everything else is optional:

```python
from pycauset import spacetime

@spacetime.register("my_diamond_4d")
class MyDiamond4D(spacetime.Spacetime):
    dimension = 4
    signature = (1, 3)                     # Lorentzian

    def sample(self, rng, n):              # (n, d) ndarray, uniform in volume measure
        return rng.uniform(0, 1, size=(n, 4))

    def is_causal(self, u, v):             # strict partial order (transitive)
        return all(u[i] < v[i] for i in range(4))

    def volume(self):
        return 1.0
    # optional: to_embedding(), boundary(), scalar_coeffs(), batch hooks
```

### Rung 2, Batch hooks (the fast path, opt-in)

If present, `sample_batch` / `is_causal_batch(coords)` (or `causality_matrix`) let the sprinkler
run the O(n²) pairwise step in NumPy/C instead of a Python loop. Same contract, better speed.
The element-wise rung stays available for small-n and prototyping.

### Code generation + online tool (the *fun* path)

The builder's recipe can be **emitted as ready-to-paste Python**, the *same template* that
`create()` uses:

```python
code = spacetime.export_python(st)        # -> a paste-ready Spacetime subclass
```

The **online generator** the director proposed is then just a thin web front-end over this exact
template: a form (dimension, signature, domain, metric, parameters) → the same code string.
Because the template is a library feature first, the website is optional and always in sync.

- **Hosting:** a static site (HTML + JS running the template client-side) is sufficient, no
  backend, no secrets, trivial to host. A small FastAPI app is the fallback if we want server-side
  validation. Recommend static + JS for R2.
- **No guessing in the tool:** every box maps to an explicit parameter; fields we can't safely
  default are left blank and *required*, with an inline note why.

---

## 6. Modify & compose spacetimes (director decision #6)

Two sanctioned ways to get a "slightly different" spacetime from a library one:

1. **Subclass** a library spacetime and override one method (e.g. tweak `is_causal`).
2. **Compose** with thin decorators that wrap a base spacetime:
   - `RestrictedSpacetime(base, region)`, cut out a subregion
   - `ConformalSpacetime(base, factor)`, apply a conformal factor
   - `TransformedSpacetime(base, transform)`, apply a coordinate transform
   - `PeriodicSpacetime(base, ...)`, impose periodic boundary conditions

Decorators compose (`Restricted(Conformal(DeSitter(d), f), region)`) and keep the contract.
Any custom or modified spacetime is **saveable/loadable** via the name registry + a serialized
*recipe* (base + parameters + transforms), so a modified `DeSitter` round-trips across sessions.

---

## 7. Fields & coefficients, never guess (director decision #3)

- The `(a, b)` derivation moves **onto the spacetime**: `Spacetime.scalar_coeffs(mass, density)`.
- Built-in Minkowski spacetimes implement the known 2D/4D table (see the Field Theory guide).
  Library curved spacetimes ship coefficients **only where documented/proven**.
- Everywhere else → `NotImplementedError`, and `propagator(a=..., b=...)` accepts manual values.
- **No name-sniffing, no heuristics, ever.** The library's known values are authored data, not
  inference. This is the "professional, not Apple" principle made structural.

---

## 8. Coordinates & embedding, hybrid model (director decision #1)

- **Default:** regenerate coordinates from the `(spacetime, seed)` provenance (memory-light,
  deterministic, today's behavior).
- **Opt-in:** attach an explicit `Embedding` (user-supplied points, adaptive/non-Poisson
  sprinklings, or post-processed coordinates).
- **Documentation must be unambiguous** about which mode an instance is in, and that a causet with
  no provenance *and* no attached embedding cannot produce coordinates (clear error, not a guess).

---

## 9. Validation, invariants (director decision #5, in plain terms)

Before physics runs on a causal matrix, we **check it is a real partial order**:

- **reflexive-free**, no element is its own cause (`C[i,i] == 0`),
- **antisymmetric**, never both `i ≺ j` and `j ≺ i`,
- **transitive**, if `i ≺ j` and `j ≺ k` then `i ≺ k` (the matrix is the *closure*, not just links).

Why: a non-transitive matrix silently produces **wrong propagators**, the worst possible failure
for research software. So we validate **eagerly** at construction and on `load()`, with a
documented `validate=False` escape hatch for experts who want the speed and accept the risk.

---

## 10. Visualization (director decision #7)

- **Backend: Plotly only.** No matplotlib focus (a static renderer may be revisited much later,
  but it is not an R2 goal).
- **Call surface, hybrid (director decision #16):** `CausalSet` methods
  (`c.plot_embedding()`, `c.plot_hasse()`, `c.plot_causal_matrix()`) for the primary citizen -
  zero imports, tab-complete, lazy plotly import in-body; a few lazy top-level `pc.plot_*`
  functions for non-causet objects (e.g. `pc.plot_heatmap(K_R)`, matrices/fields get no plot
  methods); and the fun one-verb `pc.show(c)` (plot + `.show()`). Explicit verbs only, no
  `plot(kind=…)` smart dispatch. Returns a Plotly `Figure` directly.
- **Large-set policy (uniform across `plot_embedding`, `plot_hasse`, `plot_causal_matrix`):**
  - Above a size threshold, plot a **random, seeded subset** of elements.
  - Emit a **`PyCausetPerformanceWarning`** stating what was sampled, how many elements are shown,
    and naming the bypass parameter.
  - **Bypass:** `force=True` (or `max_points=None`) renders everything, at the user's own risk.
  - Never silently truncate, always warn, always seed, always name the bypass.
- **Hasse diagrams** get the same policy; edges (the real cost) are capped alongside nodes.
- **Higher-dimensional visualization is deferred and lowest priority** (director decision #2) -
  this is a *separate* discussion for later. Candidate approaches to revisit: time-slicing,
  lightcone slices, and dimension reduction (PCA/t-SNE/UMAP) for the "point cloud" view.
- **Authored visualization (never guessed):** the spacetime *declares* how it wants to be shown -
  optional `to_embedding(coords)` (internal → 2D/3D display), `boundary()` (the shape to overlay),
  and `display_axes` (which axis is time). The viz layer has **zero geometry-specific code**:
  diamonds/cylinders/boxes "just work" because *they* declare their shapes; a custom spacetime with
  no declaration gets an honest generic fallback (raw coordinates, neutral labels, no boundary) -
  never an inferred shape. The boundary (O(1) segments) always renders, even under subsampling.
  Dimension reduction (PCA/UMAP) is *user-requested*, never auto-applied. Boundary renders only
  when points come from the spacetime's own `to_embedding`; an attached explicit `Embedding`
  suppresses/dims it (avoid a misleading shape). Mirrors `scalar_coeffs`: authored, or explicit
  fallback.
- **What "datashader" means** (for the record): it's a Python library that renders huge datasets
  (millions–billions of points) by *rasterizing* them into a pixel grid, it aggregates points per
  pixel and colors by density, instead of drawing one graphics primitive per point. Plotly draws
  each point as an object, which is why it struggles past ~10⁵ points. Datashader is the natural
  tool if R2 later needs to *render* N ~ 10⁶ embeddings; it produces a static image (less
  interactive than Plotly), so it's a *possible future addition*, **not** an R2 goal.

---

## 11. Decision log (locked)

| # | Decision | Resolution |
| :- | :-- | :-- |
| 1 | Coordinate model | **Hybrid**, regenerate by default, attach explicit embedding when asked; documented clearly |
| 2 | Dimensional scope | **Arbitrary dimension** for sprinkling/geometry/causality; viz 2D/3D required, higher-D deferred (lowest priority) |
| 3 | Coefficients | **Never guess** from a name; library-authorized values only; otherwise manual `a, b` |
| 4 | Primary object | **Causal set is primary**; spacetime/embedding are provenance/attachments |
| 5 | Validation | **Eager** partial-order validation, with `validate=False` escape hatch |
| 6 | Spacetime persistence | **Save/load + modifiable** (subclass/compose); registry + recipe serialization |
| 7 | Plotting | **Plotly only**; subset + warning + bypass for huge sets |
| 8 | Signature | **First-class**; causal order only for Lorentzian; Euclidean = point process, no causet |
| 9 | Curved set for R2.1 | **Approved: dS + AdS + FLRW** (conformally-flat = stretch) |
| 10 | Signature convention | **Blessed: `(t, s)` = (timelike, spacelike)**, Lorentzian = `(1, d−1)` |
| 11 | Registry collisions | **Agreed: explicit error + `overwrite=True`** (no silent last-wins) |
| 12 | Continuum QFT scope | **Bare-minimum Minkowski MVP only** (closed-form Green's/Wightman + `.at(coords)`); full continuum tool = future TODO, **not an R2 element** |
| 13 | Wightman / Sorkin–Johnston | **R2.0 goal (flagship)**, on flat Minkowski causets |
| 14 | Dynamics | **Yes, add bucket, deferred** (BDG action, growth models, path sum) |
| 15 | Field model | **`Field` → `CorrelatedField` → `State`**; `phi.on(causet\|spacetime)`; `pc.field("scalar", …)` string factory = sugar (unknown strings raise) |
| 16 | Visualization call surface | **Hybrid**: `CausalSet.plot_*()` methods + lazy top-level `pc.plot_*` for non-causets + `pc.show(c)` sugar; explicit verbs, no `kind=` dispatch; returns Plotly `Figure` |
| 17 | Engine/optimization scope | **Folded into R2**, the R1 post-R1 optimization program (≥0.90× NumPy, GPU parity, streaming-everything, SRP-2, eigen-cache) is tracked as the R2E phase of `R2_ROADMAP.md`, not deferred past R2 |

---

## 12. Roadmap (tentative phases)

**R2.0, Foundations + the field core (P0)**
Signature model; `Spacetime` ABC + registry + persistence; hybrid embedding; eager validation;
Tier-0 builder + Tier-1 subclass + composition decorators; Plotly subset/warning/bypass;
**Wightman / Sorkin–Johnston + propagators K_R/K_A + Pauli–Jordan iΔ** (the field core, on flat
Minkowski causets), with sign/scale conventions pinned against the continuum limit.

**R2.1, Spacetimes & remaining physics (P0/P1)**
The whole spacetime library in one phase: arbitrary-d Minkowski family, dS/AdS/FLRW
(conformally-flat stretch), black holes (stretch); spacetime-owned `scalar_coeffs`; correlators,
vevs, and causal-structure methods (see `R2_FIELDS_PHYSICS.md`).

**R2.2, Advanced physics & polish (P1/P2)**
Dimension estimators, entanglement entropy; Tier-2 batch hooks; synthetic/test posets;
final QA audit + benchmark suite. (Docs + tests are in-step in every phase, never deferred.)
Fermions/path integrals stay experimental.

**Beyond R2 (bonus)**
The Spacetime Studio website (`export_python` + hosted form); multi-time causality; JIT sampling;
datashader large-N rendering; fermions, gauge fields, interacting QFT; **causal-set dynamics (BDG
action, growth models, path sum)**; **full continuum-QFT comparison tooling** (R2 ships only the
bare-minimum Minkowski MVP).

---

## 13. Remaining open questions (bring to the director)

1. **Higher-dimensional visualization**, which technique(s) do we adopt for d > 3, and when?
   (Deferred, but worth a dedicated design session.)
2. **Studio launch modes**, both are specced (GitHub Pages static page, and `pycauset.studio()`
   open-in-browser à la Plotly). Decide which ships first, or ship both.
3. **"Extensive" bar**, how far to push the library before R2 ships vs after (black holes,
   multi-time, synthetic models)?

---

## 14. Engine & optimization (folded into R2)

The optimization work deferred out of R1 is **part of R2**, not a separate post-R2 effort
(decision #17). It is tracked as the **R2E** phase in [`R2_ROADMAP.md`](R2_ROADMAP.md), with
per-op status in `documentation/internals/plans/OPTIMIZATION_STATUS.md`:

- **R2_PERF**, ≥ 0.90× NumPy throughput for every op, CI-enforced.
- **R2_CPU**, modern tiled CPU engine (absorbs `archive/R1_CPU_PLAN.md`).
- **R2_GPU**, GPU parity or explicit support status (absorbs `archive/R1_GPU_PLAN.md`).
- **R2_STREAM**, streaming/out-of-core across the op surface.
- **R2_CATALOG**, SRP-2 "Causal Math Optimization Catalog".
- **R2_EIGCACHE**, eigen-cache persistence to `.pycauset`.
- **R2_HARDEN**, post-R1 bug/polish backlog.

The engine track runs in parallel with the physics phases and feeds `R2_QA` (the final gate).

**Current R2E status (2026-08-29):** R2.1 ships the safe engine items, `R2_EIGCACHE`, elementwise
f64 SIMD + view-hardening (R2_CPU), 3/4 SRP-2 shortcuts (R2_CATALOG), and the parity baseline +
matmul-OpenBLAS root-cause (R2_PERF). R2.2 takes the larger-risk remainder: the `≥ 0.90× NumPy`
bar, the lazy-elementwise routing (reverted, stack overrun, see BUG_LOG), the SRP-2 skew
eigensystem, R2_GPU (VS 2022 + CUDA 12.6), R2_STREAM, and R2_HARDEN.
