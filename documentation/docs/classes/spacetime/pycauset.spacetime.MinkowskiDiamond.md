# pycauset.spacetime.MinkowskiDiamond

```python
class pycauset.spacetime.MinkowskiDiamond(dimension: int = 2)
```

The causal diamond (Alexandrov interval) between $p = (0, \vec 0)$ and $q = (1, \vec 0)$ in
$d$-dimensional Minkowski space ($c = 1$): the points with $|\vec x| < \min(t, 1 - t)$.
Coordinates are $(t, x_1, \dots, x_{d-1})$.

Sprinkling is uniform in the diamond (rejection sampling from its bounding box), and
$u \prec v$ exactly when $t_v > t_u$ and $(t_v - t_u)^2 > |\vec x_v - \vec x_u|^2$. One native
implementation (`include/pycauset/causet/Spacetime.hpp`) does both.

## Parameters

*   **dimension** (*int*): $d \ge 2$ (one time and $d - 1$ space dimensions).

## Methods

*   **dimension()** → *int*
*   **volume()** → *float*: $2 V_{d-1} (1/2)^d / d$, with $V_k$ the volume of the unit $k$-ball
    ($1/2$ in $d = 2$). A sprinkle at density $\rho$ has $\rho \cdot$ `volume()` points on average.
*   **is_causal(u, v)** → *bool*: whether point `u` is in the causal past of point `v`.

## Example

```python
import pycauset as pc

st = pc.spacetime.MinkowskiDiamond(4)
c = pc.causet(n=2000, spacetime=st, seed=1)
c.myrheim_meyer_dimension()     # about 4
```
