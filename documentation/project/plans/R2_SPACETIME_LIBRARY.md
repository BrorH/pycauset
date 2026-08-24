# R2 Spacetime Library — Catalog (Spec)

**Status**: Planning. Defines *what* spacetimes R2 ships, and the contract each must satisfy.
**Companion**: `R2_PLAN_MAP.md` (feature map), `R2_SPACETIME_CREATION.md` (how to add new ones).

---

## 1. The contract every library spacetime must satisfy

A library spacetime is just a `Spacetime` that ships in-tree, documented and tested. It must
provide exactly the same contract as a custom one (see `R2_SPACETIME_CREATION.md`), plus:

1. **`dimension` and `signature`** declared explicitly (never implied).
2. **`volume()`** exact, and equal to the total mass of the measure `sample()` draws from.
3. **`sample(rng, n)`** uniform w.r.t. that measure, deterministic given the injected RNG.
4. **`is_causal(u, v)`** a strict partial order that is **transitive** (the closure, not links).
5. **`scalar_coeffs(mass, density)`** present **only if** the coefficients are documented/proven;
   otherwise it raises `NotImplementedError`. No guessed coefficients, ever.
6. Optional `to_embedding()` / `boundary()` for plotting.

**Definition of done** for each entry: exact `volume`; sampler uniform (Monte Carlo test vs
formula); order validated transitive/irreflexive/antisymmetric; reproducibility test (same seed ⇒
bit-identical); docs page + nav entry. Curved entries also get a "known coefficients: none —
manual `a, b`" note unless we actually derive them.

---

## 2. Catalog

Legend: `d` = total dimension, `t` = timelike dims, `s` = spacelike dims (signature `(t, s)`).
Priorities: P0 = R2.0, P1 = R2.1, P2 = R2.2/later.

### 2.1 Flat Minkowski family

| Spacetime | dim / sig | Coordinates | Volume | Causality | Coeffs | Prio |
| :-- | :-- | :-- | :-- | :-- | :-- | :-: |
| `MinkowskiDiamond(d)` | d / (1, d-1) | 1+1: lightcone `(u,v) ∈ [0,1]²` | 1 (normalized) in 1+1 | `u_p<u_q ∧ v_p<v_q` | 2D/4D known | P0 |
| `MinkowskiCylinder(d, height, circumference)` | d / (1, d-1) | `(t, x)` with `x` periodic | `height × (spatial volume)` | `dt > wrapped_spatial_dist` | 2D/4D known | P0 |
| `MinkowskiBox(d, time_extent, space_extent)` | d / (1, d-1) | `(t, x, …)` | `T · L^{d-1}` | `dt > ‖dx‖` (Euclidean) | 2D/4D known | P0 |
| `MinkowskiSlab(d, height)` (IR/UV cutoff) | d / (1, d-1) | `(t, x)` | `height × (spatial volume, regulated)` | `dt > ‖dx‖` | manual | P1 |
| `MinkowskiCone(d)` (future lightcone) | d / (1, d-1) | `(t, x)` | finite (cone volume) | timelike future | manual | P2 |

**Current gap to fix in R2:** today's `MinkowskiDiamond(d)` for `d > 2` is a *product of
lightcone intervals* `[0,1]^d`, which is a placeholder — **not** the true d-dimensional causal
diamond (`I⁺(p) ∩ I⁻(q)`). R2 must implement the real diamond (correct volume + uniform sampler
+ causal predicate), and either rename or retire the placeholder. The `MinkowskiBox(d)` and
`MinkowskiCylinder(d)` code generalizes more honestly, but only `d=2` is exposed/tested today.

### 2.2 Curved / cosmological

| Spacetime | dim / sig | Construction | Causality | Coeffs | Prio |
| :-- | :-- | :-- | :-- | :-- | :-: |
| `DeSitter(d, R)` | d / (1, d-1) | hyperboloid `−X₀² + X₁² + … + X_d² = R²` in Minkowski_{d+1} | from the dS metric | manual (unless derived) | P1 |
| `AntiDeSitter(d, R)` | d / (1, d-1) | hyperboloid `−X₀² − X₁² + … + X_d² = −R²` | from the **universal cover** | manual | P1 |
| `FLRW(d, a(t), k)` | d / (1, d-1) | comoving `(t, x)`, `ds² = −dt² + a(t)² dΣ_k²`, `k∈{−1,0,+1}` | solve null condition | manual | P1 |
| `ConformallyFlat(d, Ω(x))` | d / (1, d-1) | flat × conformal factor | `dt² > ‖dx‖²` (same lightcone) | manual | P1 |

**Honesty notes (never-guess policy, made explicit):**
- **AdS needs its universal cover.** The naive hyperboloid contains closed timelike curves; a
  *causal* AdS spacetime is the cover. R2 must ship the cover (or clearly document the naive
  version as "no causal order"). This is exactly the kind of subtlety a "professional, not Apple"
  product must surface rather than paper over.
- **None of these ship automatic `scalar_coeffs`** unless we derive and document them. R2.1 ships
  the geometry; coefficients remain manual `a, b`. Deriving dS/AdS/FLRW coefficients is a research
  task, not a coding task — flag for a later, dedicated effort.

### 2.3 Black holes (later)

| Spacetime | dim / sig | Construction | Causality | Coeffs | Prio |
| :-- | :-- | :-- | :-- | :-- | :-: |
| `Schwarzschild(d, M)` | d / (1, d-1) | Eddington–Finkelstein (or Kruskal) patch | null geodesics in the patch | manual | P2 |
| `ReissnerNordstrom(d, M, Q)` | d / (1, d-1) | charged, spherically symmetric; two horizons | null geodesics in the patch | manual | P2 |
| `Kerr(M, a)` | 4 / (1, 3) | rotating, axially symmetric; ergosphere + ring singularity | null geodesics; CTCs inside inner horizon | manual | P2 |
| `KerrNewman(M, a, Q)` | 4 / (1, 3) | rotating + charged (most general electrovac solution) | as Kerr, plus charge | manual | P2 |

Hard samplers + horizon/ergosphere handling; explicitly P2/later, not a blocking R2 item.

**Honesty notes (never-guess policy):** Kerr contains closed timelike curves inside the inner
horizon and an ergosphere (no global time), and Reissner–Nordström has a Cauchy horizon and a
timelike singularity — richer causal pathology than Schwarzschild. All ship **geometry-only** with
manual `a, b`. Higher-dimensional rotation generalizes to **Myers–Perry** (deferred); the charged
family generalizes to any `d` more directly.

### 2.4 Compact / periodic

| Spacetime | dim / sig | Notes | Prio |
| :-- | :-- | :-- | :-: |
| `TorusT×Spatial(d, extents)` | d / (1, d-1) | periodic in all spatial dims | P1 |
| `SphereCylinder(d, R, height)` | d / (1, d-1) | `S^{d-1} × ℝ` spherical spatial slice | P2 |

These are natural targets for the `PeriodicSpacetime` decorator in `R2_SPACETIME_CREATION.md`,
so they may ship as *compositions* rather than bespoke classes.

### 2.5 Synthetic / test (order generators — "a causet is just a poset")

Causal sets are partially ordered sets; these generators build **orders directly** (no continuum),
for testing, pedagogy, and null models. They satisfy the sprinkling API surface but skip geometry.

| Generator | What it produces | Use | Prio |
| :-- | :-- | :-- | :-: |
| `Chain(n)` | the total order (fully causal, degenerate causet) | edge cases; transitive-percolation base | P2 |
| `Antichain(n)` | the empty order (no relations) | edge cases; validation | P2 |
| `TransitivePercolation(p, n)` | random causet from bond percolation on a total order | null models; pedagogy | P2 |
| `RandomDAGOrder(p, n)` | random acyclic upper-triangular edges + transitive closure | the "raw" random causet; matches bit-matrix storage | P2 |
| `KleitmanRothschild(n)` | random 3-layer poset ("almost all" posets) | asymptotics; stress tests | P2 |
| `IntervalOrder(n, …)` | intervals on the real line (the order class 1+1 sprinklings belong to) | pedagogy; structural tests | P2 |
| `Dimension2Poset(perm)` | poset of dimension 2 built from a permutation | structural tests | P2 |
| `ProductOrder(dims)` | product of chains (grid poset) | structured tests | P2 |
| `Poset(relations)` | an explicit user order wrapped as a `CausalSet` source | invariant testing | P2 |

These exercise validation and Hasse tooling without geometry, and make the point structural: the
library treats an order as first-class, independent of any continuum it may (or may not) come from.

---

## 3. Curved set for R2.1 (approved)

**Director approved: dS + AdS + FLRW** are must-have for R2.1; conformally-flat is a stretch.
Black holes (Schwarzschild, Reissner–Nordström, Kerr, Kerr–Newman) and exotic multi-time
signatures wait for R2.2/later.
