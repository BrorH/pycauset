# R2 Fields & Physics — Discussion Draft

**Status**: Field model **locked** (field → `CorrelatedField` → state). The physics feature list
remains a brainstorm to take to the causal-sets conference.
**Companion**: `R2_PLAN_MAP.md`, `R2_API_DESIGN.md`.

---

## 1. How do we define a field? (locked)

The director and I have converged on: **the field is one object; you *apply* it to a background.**

```python
phi  = pc.field("scalar", mass=1.5)   # the field (species/kind: scalar, mass 1.5) — set-independent
Q_c  = phi.on(causet)                 # a CorrelatedField on a causal set
Q_ct = phi.on(spacetime)              # a CorrelatedField on a continuum spacetime (for limits)
```

`phi.on(X)` returns a **`CorrelatedField`** — the field together with its correlation functions
(Green's functions, two-point functions) on X — **not** a "state" and **not** an "instance":

| Concept | Physics meaning | Depends on background? |
| :-- | :-- | :-: |
| **Field** (`phi`) | the field content: species, mass, spin, discretization scheme | no |
| **Correlated field** (`phi.on(X)`) | the field + its vacuum 2-point function + Green's functions on X | yes |
| **State / excitation** (`Q.state(...)`) | a specific excitation of the vacuum (1-particle, coherent, classical config) | yes |

This matches how physicists think: *fields create excitations of the vacuum — those are your
states.* The correlated field holds the vacuum (Sorkin–Johnston, on a causet); states are built on
top of it.

> **Why "CorrelatedField"?** The name is a *noun* describing what the object **is** — the field
> carrying its correlation functions on a background — rather than an *act* ("quantized": `.on()`
> does not quantize) or a *restriction* ("free": interactions would break it). It also survives
> interactions, which merely add more correlators. (Director's choice.)

**One uniform API for causet and continuum.** Both `Q_c` and `Q_ct` expose the same verbs:

```python
Q.propagator()            # retarded Green's function
Q.propagator(advanced=True)
Q.pauli_jordan()          # iΔ (commutator function)
Q.wightman()              # vacuum 2-point (SJ on a causet; continuum Wightman on a spacetime)
Q.correlator()            # ⟨φφ⟩
```

Differences are honest and typed:

- **Causet** → the propagator/Wightman are `N×N` matrices (indexed by elements); `(a, b)` derive
  from the causet's density ρ = n/V.
- **Continuum** → the Green's functions are *kernels* `G(x, y)` (callables of two points); no
  density. To compare against the discrete matrix, sample the kernel at the causet's coordinates:
  `Q_ct.at(coords)`.

**State-independence (the honest semantics):** `K_R`, `K_A`, and `iΔ` are *state-independent* —
determined by field + background alone. The Wightman function `W` is *state-dependent*: it
requires choosing a **vacuum** (the Sorkin–Johnston prescription, applied inside `.wightman()` by
default). That choice — not the act of `.on()` — is the "quantization" step. This is why it is a
*correlated* field, not a *quantized* one.

**Continuum scope (director's decision — bare minimum):** continuum comparison is an **MVP-only**
feature in R2: flat **Minkowski (1+1 and 3+1)** closed forms for the retarded Green's function and
Wightman function, plus `Q_ct.at(coords)` sampling — just enough to run a continuum-limit sanity
check. A *full* continuum-QFT tool (arbitrary curved backgrounds, interacting fields,
renormalization) is a **future TODO, officially not an R2 element.** `.on(spacetime)` for anything
else raises `NotImplementedError` — never guess, and don't scope-creep into QFT.

**Fermions (honest):** spin on a causal set is an open research area (no clean Dirac operator from
the causal structure yet). Ship scalar physics flawlessly; reserve fermions as an explicitly
experimental module.

### Decisions locked

- **Returned object = `CorrelatedField`** — a noun naming what the object *is* (the field carrying
  its correlations on a background), not an act ("quantized") or a restriction ("free").
- **String factory** `pc.field("scalar", mass=…)` = sugar over explicit classes; unknown strings
  raise.

---

## 2. Physics feature brainstorm (for the conference)

### Core — must work flawlessly (P0/R2.0)

| Feature | What it is | Notes |
| :-- | :-- | :-- |
| Retarded propagator K_R | `K_R = aC(I − baC)⁻¹` | pin sign/factor conventions + test vs continuum |
| Advanced propagator K_A | `K_A = K_Rᵀ` | trivial but must be exact |
| Pauli–Jordan iΔ | `Δ = K_R − K_A`; store antisymmetrically with scalar `1j` | verify antisymmetry |
| **Sorkin–Johnston vacuum / Wightman** | `iΔ` is Hermitian; `W =` its positive-eigenvalue part | flagship — needs `eigh` (already in linalg) |
| 2-point correlator ⟨φφ⟩ | from Wightman | free-field Wick for higher points |
| vevs ⟨φ⟩, ⟨φ²⟩ | field configuration + measure | document UV regularization caveats |

### Causal-structure methods (on `CausalSet`, not `Field`)

| Feature | Notes |
| :-- | :-- |
| links / Hasse (`transitive reduction`) | already used by viz; expose first-class |
| chains, antichains, longest chain | longest chain ≈ geodesic length |
| intervals `I(x,y) = future(x) ∩ past(y)` | the Alexandrov interval — feeds SJ + dimension |
| past/future sets, layering | structural analysis |
| **dimension estimators** | Myrheim–Meyer, mid-point scaling, spectral dimension (from d'Alembertian eigenvalues) — very "fun" |

### Advanced (P1/R2.2)

| Feature | Notes |
| :-- | :-- |
| **Entanglement entropy / mutual information** | Sorkin–Yazdi, from the Wightman function — needs SJ first |
| geodesic / timelike-distance estimators | Myrheim–Meyer, longest-chain |
| manifoldlikeness tests | valency, interval abundance |
| continuum-limit harness (Minkowski MVP) | `phi.on(spacetime)` → `Q_ct.at(coords)` diff vs `Q_c.wightman()`; full continuum tool = future |

### Research-grade (honest — P2/later)

| Feature | Notes |
| :-- | :-- |
| Fermions (Dirac operator) | experimental module; open research |
| vector / gauge fields | later |
| interacting fields / path integrals | framework + measure; perturbative amplitudes |
| spectral geometry (heat-kernel traces, spectral action) | beyond scalar |

### Dynamics (deferred — future, not R2)

| Feature | Notes |
| :-- | :-- |
| Causal-set action (BDG) | Benincasa–Dowker–Glaser d'Alembertian → Einstein–Hilbert limit |
| Growth models / path sum | classical sequential growth; sum-over-histories |

---

## 3. Sign/scale conventions (a core-correctness task, not a design choice)

The exact factors in `K_R`, `iΔ`, and `W` depend on metric convention, the `a`/`b` table, and the
`1j` storage trick. R2.1 must **pin these down and test them against known continuum results**
(free scalar in 1+1 and 3+1 Minkowski). This is what "flawless" means — not just that it runs, but
that it reproduces the continuum limit.
