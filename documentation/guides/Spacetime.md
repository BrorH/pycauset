# Spacetime Manifolds

`pycauset` provides a library of standard spacetime manifolds that can be used as the domain for sprinkling causal sets. These are available in the `pycauset.spacetime` module.

## Available Spacetimes

### MinkowskiDiamond

The `MinkowskiDiamond` represents a causal diamond in flat Minkowski space. This is the intersection of the future lightcone of a point $p$ and the past lightcone of a point $q$.

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

The `MinkowskiCylinder` represents a flat Minkowski spacetime with periodic boundary conditions in the spatial dimension. This topology is $S^1 \times \mathbb{R}$ (circle $\times$ time).

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

## Using Spacetimes with CausalSet

You can pass these spacetime objects to the `CausalSet` constructor.

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
