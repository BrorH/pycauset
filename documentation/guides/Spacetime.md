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
c = pycauset.Causet(n=1000, spacetime=st)
```

### Poisson Sprinkling (Density)

Instead of specifying $N$, you can specify a sprinkling `density` $\rho$. The number of points $N$ will be drawn from a Poisson distribution:
$$ N \sim \text{Poisson}(\rho \times V) $$
where $V$ is the volume of the spacetime region.

```python
# Sprinkle with density 100 points per unit volume
# Total volume = 50, so expected N = 5000
c = pycauset.Causet(density=100, spacetime=st)
```
