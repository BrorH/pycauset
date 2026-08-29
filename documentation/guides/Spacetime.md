# Spacetime Manifolds

`pycauset` provides a library of standard spacetime manifolds that can be used as the domain for sprinkling causal sets. These are available in the [[pycauset.spacetime]] module.

## Available Spacetimes

### MinkowskiDiamond

The [[pycauset.spacetime.MinkowskiDiamond]] represents a causal diamond in flat Minkowski space. This is the intersection of the future lightcone of a point $p$ and the past lightcone of a point $q$.

**Coordinates**:
*   **2D (1+1)**: Uses **Lightcone Coordinates** $(u, v)$ where $u, v \in [0, 1]$.
    *   Metric: $ds^2 = -du dv$ (up to a factor of 2 depending on convention).
    *   Causality: $p \prec q \iff u_p < u_q \text{ AND } v_p < v_q$.
    *   Volume: Normalized to $1.0$ in these coordinates.


```python
from pycauset import spacetime

# Create a 2D Minkowski Diamond
diamond = spacetime.MinkowskiDiamond(dimension=2)
```



### MinkowskiCylinder

The [[pycauset.spacetime.MinkowskiCylinder]] represents a flat Minkowski spacetime with periodic boundary conditions in the spatial dimension. This topology is $S^1 \times \mathbb{R}$ (circle $\times$ time).

**Coordinates**:
*   **2D (1+1)**: Uses **Standard Coordinates** $(t, x)$.
    *   $t \in [0, \text{height}]$
    *   $x \in [0, \text{circumference})$
    *   Causality: $t_2 > t_1$ AND $(t_2 - t_1) > \text{shortest\_dist}(x_1, x_2)$ on the circle.
    *   Volume: $\text{height} \times \text{circumference}$.

```python
from pycauset import spacetime

# Create a cylinder with height 2.0 and circumference 3.0
cylinder = spacetime.MinkowskiCylinder(dimension=2, height=2.0, circumference=3.0)
```

### MinkowskiBox

The [[pycauset.spacetime.MinkowskiBox]] represents a rectangular block in flat Minkowski space with "hard wall" boundaries. This is useful for studying boundary effects where the boundaries are not null surfaces (unlike the Diamond).

**Coordinates**:
*   **2D (1+1)**: Uses **Standard Coordinates** $(t, x)$.
    *   $t \in [0, \text{time\_extent}]$
    *   $x \in [0, \text{space\_extent}]$
    *   Causality: Standard Minkowski causality $\Delta t > |\Delta x|$.
    *   Volume: $\text{time\_extent} \times \text{space\_extent}$.

```python
from pycauset import spacetime

# Create a box with T=2.0 and L=1.0
box = spacetime.MinkowskiBox(dimension=2, time_extent=2.0, space_extent=1.0)
```

## Visualization Support

All standard spacetimes support the visualization interface used by [[docs/pycauset.vis/index.md|pycauset.vis]]. They implement:

*   `transform_coordinates(coords)`: Converts internal coordinates (like lightcone $u,v$) to visualization-friendly coordinates (like Cartesian $t,x$ or 3D Cylindrical).
*   `get_boundary()`: Returns the geometry of the spacetime boundary for plotting.

See the [[guides/Visualization|Visualization Guide]] for more details.

## Using Spacetimes with CausalSet

You can pass these spacetime objects to the [[docs/classes/spacetime/pycauset.CausalSet.md|pycauset.CausalSet]] constructor.

### Fixed Number Sprinkling

Sprinkle exactly $N$ points into the spacetime.

```python
import pycauset
from pycauset import spacetime

st = spacetime.MinkowskiCylinder(2, height=10, circumference=5)
c = pycauset.causet(n=1000, spacetime=st)
```

### Poisson Sprinkling (Density)

Instead of specifying $N$, you can specify a sprinkling `density` $\rho$. The number of points $N$ will be drawn from a Poisson distribution:
$$ N \sim \text{Poisson}(\rho \times V) $$
where $V$ is the volume of the spacetime region.

```python
# Sprinkle with density 100 points per unit volume
# Total volume = 50, so expected N = 5000
c = pycauset.causet(density=100, spacetime=st)
```

## Defining Custom Spacetimes (R2)

`pycauset.spacetime.Spacetime` is the abstract base class that makes custom spacetimes
first-class. Subclass it and implement four methods, `dimension()`, `volume()`,
`sample(rng, n)`, and `is_causal(u, v)`, and you get a spacetime the sprinkler,
field engine, and visualizer all understand.

### Signature

Every spacetime has a **signature** `(t, s)` = (timelike, spacelike). It defaults to
Lorentzian `(1, d-1)`; declare `signature = (t, s)` as a class attribute to override.
A causal order (`is_causal`) exists only for Lorentzian `t == 1`: Euclidean `(0, d)`
spacetimes are point processes, and multi-time `(t > 1, s)` spacetimes must supply
their own "future" convention, the base `is_causal` raises rather than guessing.

```python
from pycauset import spacetime

@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    def dimension(self):
        return 2

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2))

    def is_causal(self, u, v):
        return u[0] < v[0] and u[1] < v[1]
```

### Registry

`@spacetime.register("name")` gives a spacetime a persistable name (used by
save/load). Duplicate names raise unless you pass `overwrite=True`.
`spacetime.create(...)` (Rung 0) assembles a spacetime from a declarative recipe -
currently the flat Minkowski family (`domain="diamond" | "cylinder" | "box"`).

Optional hooks: `scalar_coeffs(mass, density)` (authored field coefficients, the
default raises, never guesses), `is_causal_batch(coords)` (fast path),
`to_embedding(coords)` and `boundary()` (presentation).

### Composition decorators (R2_CREATE)

Build new spacetimes by wrapping an existing one instead of writing a fresh
subclass. Each decorator keeps `volume ↔ sample` consistent:

*   **[[docs/classes/spacetime/pycauset.spacetime.RestrictedSpacetime.md|RestrictedSpacetime]]**, keep a
    subregion selected by a predicate; rejection sampling + Monte-Carlo volume.
*   **[[docs/classes/spacetime/pycauset.spacetime.TransformedSpacetime.md|TransformedSpacetime]]**, apply a
    (volume-preserving) coordinate transform `forward`/`inverse`.
*   **[[docs/classes/spacetime/pycauset.spacetime.ConformalSpacetime.md|ConformalSpacetime]]**, a conformal
    factor `Omega(x)` preserves the light-cone but rescales the volume measure by `Omega^d`.
*   **[[docs/classes/spacetime/pycauset.spacetime.PeriodicSpacetime.md|PeriodicSpacetime]]**, periodic
    identification along spacelike axes (periodic time raises: it would create CTCs).

```python
from pycauset import spacetime

box = spacetime.MinkowskiBox(2, 10.0, 10.0)
half  = spacetime.RestrictedSpacetime(box, region=lambda c: c[1] < 5.0)
blown = spacetime.ConformalSpacetime(box, conformal_factor=lambda c: 2.0)
ring  = spacetime.PeriodicSpacetime(box, periods={1: 5.0})
```

### Curved spacetimes (R2_CURVED)

`spacetime.DeSitter`, `spacetime.AntiDeSitter`, and `spacetime.FLRW` ship as
documented **parametrizations** (their samplers are not the invariant measure, and
`scalar_coeffs` raises, coefficients are manual `a, b`). `DeSitter` carries the
ambient-Minkowski causal order; `AntiDeSitter` is flagged "no causal order" (the
naive hyperboloid has closed timelike curves); `FLRW` uses the null-geodesic order.
`Schwarzschild` (1+1) uses the exact radial tortoise null condition; the other black
holes are parked.

### Sprinkling a custom spacetime + validation

A custom spacetime sprinkles through the same `CausalSet` API. Points are labelled
by time (coordinate index 0) so the stored matrix is strictly upper-triangular, and
the sampled coordinates are attached as an embedding (served by `coordinates()`).

```python
import pycauset
from pycauset import spacetime

@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    def dimension(self): return 2
    def volume(self): return 1.0
    def sample(self, rng, n): return rng.uniform(0.0, 1.0, size=(n, 2))
    def is_causal(self, u, v): return u[0] < v[0] and u[1] < v[1]

c = pycauset.causet(n=500, spacetime=MyDiamond(), seed=42)
c.validate()      # verifies the order is a strict partial order
c.coordinates()   # the attached embedding (500, 2)
```

`CausalSet(matrix=..., validate=True)` (the default) rejects a matrix that is not
reflexive-free, antisymmetric, and transitive; pass `validate=False` to skip the check.
