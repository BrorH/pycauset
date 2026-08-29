# R2 Feature Menu

A one-page map of what Release 2 ships, the "conference feature menu". Each item is
a shippable capability with a minimal demo recipe; follow the link for the full guide.

## Spacetimes (R2_SIG / R2_ABC / R2_MINK / R2_CURVED / R2_CREATE)

*   **Flat family**, `MinkowskiDiamond` (2D causal diamond), `MinkowskiCylinder`
    (2D), `MinkowskiBox` (arbitrary `d`), each with exact volume, a uniform sampler,
    and a causal predicate.
*   **Curved**, `DeSitter`, `AntiDeSitter`, `FLRW`, `Schwarzschild` (1+1) ship as
    documented parametrizations (honest caveats; manual field coefficients).
*   **First-class signature**, every spacetime declares `(t, s)`; no hidden
    Lorentzian guess.
*   **Composition decorators**, `Restricted`, `Transformed`, `Conformal`, `Periodic`
    build new spacetimes from existing ones.

```python
from pycauset import spacetime as sp
diamond = sp.MinkowskiDiamond(2)
half    = sp.RestrictedSpacetime(diamond, region=lambda c: c[1] < 0.5)
ring    = sp.PeriodicSpacetime(diamond, periods={1: 1.0})
```

*   **Declarative + codegen**, `spacetime.create(recipe)` and
    `spacetime.export_python(...)`.

## Causal sets (R2_VALIDATE / R2_EMBED / R2_STRUCT)

*   **Eager validation**, a non-transitive matrix is rejected at construction.
*   **Structure methods**, `links()`, `past()`, `future()`, `interval()`,
    `is_chain()`, `is_antichain()`, `longest_chain()`, `layers()`.
*   **Coordinates**, sampled embeddings served by `coordinates()`.

```python
import pycauset as pc
c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=42)
c.validate()                 # strict partial order guaranteed
links = c.links()            # transitive reduction
```

## Fields, the Sorkin–Johnston flagship (R2_FIELD / R2_KRD / R2_SJ)

*   `pc.field("scalar", mass=…)` → a set-independent `Field`; `phi.on(causet)` returns
    a `CorrelatedField`; `phi.on(spacetime)` returns a continuum field (massless 1+1).

```python
phi = pc.field("scalar", mass=0.0)
Q = phi.on(c)                # CorrelatedField
KR = Q.retarded()            # K_R
iD = Q.pauli_jordan()        # iΔ = K_R − K_A
W  = Q.wightman()            # Sorkin–Johnston vacuum two-point (positive part of iΔ)
S  = Q.entanglement_entropy(region=[0, 1, 2])
```

## Dimension, entropy, synthetic data (R2_DIM / R2_ENT / R2_SYNTH)

```python
d = c.myrheim_meyer_dimension()   # Myrheim–Meyer dimension estimator
import pycauset.synthetic as syn
chain = syn.chain(20)             # also antichain, transitive_percolation, …
```

## Visualization (R2_VIZ)

```python
c.plot_embedding()    # or c.plot_hasse(), c.plot_causal_matrix(); pc.show(c)
```

Authored spacetime shapes drive the plot; geometry-free spacetimes render raw; large
sets warn + subset (`force=True` bypasses).

## See also

*   [[guides/Spacetime|Spacetime guide]] · [[guides/Field Theory|Field Theory guide]]
    · [[guides/Visualization|Visualization guide]]
*   [[docs/index|API Reference]] · [[project/plans/R2_ROADMAP.md|R2 Roadmap]]
