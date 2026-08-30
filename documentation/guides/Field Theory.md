# Field Theory on Causal Sets

Causal set theory is a framework for quantum field theory as much as for spacetime
itself. PyCauset keeps the two concerns separate: the geometry is a
[[pycauset.CausalSet]], the matter is a [[pycauset.field.Field]]. You define a field
once and apply it to whatever background you like.

## The field API

A **field** is a set-independent object (species + mass). Applying it to a background
returns a **correlated field** that knows the Green's functions on that background.

```python
import pycauset as pc

phi = pc.field("scalar", mass=1.5)   # the Field, background-independent
Q   = phi.on(c)                      # a CorrelatedField: the field on this causet
```

`Q` exposes the free-field core:

```python
K_R = Q.retarded()      # retarded Green's function, K_R = aC (I - baC)^-1
K_A = Q.advanced()      # advanced Green's function, K_A = K_R^T
iD  = Q.pauli_jordan()  # iΔ = K_R - K_A  (the Hermitian commutator function)
W   = Q.wightman()      # Sorkin-Johnston vacuum two-point function (positive part of iΔ)
G   = Q.correlator()    # ⟨φφ⟩ = W (free field)
```

`pauli_jordan()` returns the real matrix $\Delta$ with a scalar factor of $i$ attached
rather than storing complex numbers twice; the factor is applied automatically when
you read elements or convert to NumPy.

`Q.state()` returns the vacuum `State`. `Q.propagator()` is an alias for the retarded
propagator.

### Entanglement entropy

```python
S = Q.entanglement_entropy(region)                       # Sorkin-Yazdi "1/2" convention
S = Q.entanglement_entropy(region, convention="symplectic")  # literal symplectic form
```

Two conventions are available (see
[[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]]): the default
`"sorkin_yazdi"` absorbs the $1/2$ zero-point so it is well defined for the
Sorkin-Johnston Wightman $W \ge 0$, and `"symplectic"` is the literal form, which
requires $W \ge 1/2$.

### Continuum limit

`phi.on(spacetime)` (a continuum Minkowski spacetime) returns a
`ContinuumCorrelatedField` with closed-form Green's functions and a `.at(coords)`
sampler, for comparing the discrete theory to the continuum.

## The retarded propagator and the coefficients

The retarded propagator is the inverse of the causal-set d'Alembertian:

$$ K_R = \Phi (I - b\Phi)^{-1}, \qquad \Phi = aC $$

where $C$ is the causal matrix. The two coefficients $a$ and $b$ connect the discrete
matrix to the continuous physics; they depend on dimension $d$, sprinkling density
$\rho = N/V$, and mass $m$.

PyCauset derives them from the spacetime itself (`Spacetime.scalar_coeffs`), never
guessing. For flat Minkowski spacetimes:

| Dimension | $a$ | $b$ |
| :--- | :--- | :--- |
| 2D | $1/2$ | $-m^2/\rho$ |
| 4D | $\sqrt{\rho} / (2\pi\sqrt{6})$ | $-m^2/\rho$ |

Curved spacetimes (`DeSitter`, `AntiDeSitter`, `FLRW`, `Schwarzschild`) ship as
documented parametrizations whose `scalar_coeffs` raises; you pass $a$ and $b$ manually
for those.

### Massless limit

For $m = 0$, $b = 0$ and the propagator simplifies to $K_R = aC$.

## Using density-based sprinkling

The coefficients need a density, so sprinkle with `density` (or pass the density
explicitly) rather than a fixed `n`:

```python
import pycauset as pc

st = pc.spacetime.MinkowskiDiamond(2)
c = pc.causet(density=1000, spacetime=st, seed=1)

phi = pc.field("scalar", mass=1.5)
Q = phi.on(c)
K = Q.retarded()
```

## Manual coefficients

To override the derived coefficients, use the legacy `ScalarField`, whose
`propagator(a=..., b=...)` accepts them directly:

```python
from pycauset.field import ScalarField

field = ScalarField(c, mass=1.5)
K = field.propagator(a=0.5, b=-0.02)
```

## Legacy ScalarField

`ScalarField` predates the `field`/`CorrelatedField` split. It is a single object
that wraps a causet plus a mass, and computes the retarded propagator and the
Pauli-Jordan function:

```python
from pycauset.field import ScalarField

field = ScalarField(c, mass=1.5)
K   = field.propagator()        # TriangularFloatMatrix
iD  = field.pauli_jordan()      # AntiSymmetricMatrix (Δ with scalar factor 1j)
```

It remains for back-compat. New code should use `pc.field(...).on(...)`, which
separates the set-independent field from the correlated field and adds the
Sorkin-Johnston Wightman vacuum, the correlator, state, and entanglement entropy.

## Scope

The shipped field core covers the free scalar field: the retarded/advanced
propagators, the Pauli-Jordan function $i\Delta$, and the Sorkin-Johnston Wightman
vacuum. Massive Green's functions with Bessel kernels, the continuum Wightman log,
higher-point Wick contractions, interacting fields, fermions, and gauge fields are
future work.

See [[docs/classes/field/pycauset.field.CorrelatedField.md|CorrelatedField]] for the
full method list.
