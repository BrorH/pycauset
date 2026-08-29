# pycauset.spacetime.Schwarzschild

```python
class Schwarzschild(dimension=2, mass=1.0, r_max=10.0, time_extent=10.0)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Schwarzschild black hole, geometry-only, exterior region `r > 2M` in Schwarzschild coordinates.

## Parameters

*   **dimension** (*int*): Spacetime dimension. Only `2` (1+1, radial) is implemented.
*   **mass** (*float*): The mass `M` (horizon at `r = 2M`).
*   **r_max** (*float*): The outer radial cutoff.
*   **time_extent** (*float*): The time range.

## Notes (honest caveats)

- `is_causal` is the **exact** 1+1 radial null condition via the tortoise coordinate `r* = r + 2M ln(r/2M − 1)`: `Δt ≥ |Δr*|`. Higher dimensions (the angular null geodesic) are a research task and raise `NotImplementedError`.
- The sampler is uniform in `(t, r)` over the exterior patch (a documented parametrization).
- `scalar_coeffs` raises `NotImplementedError`, coefficients are manual `(a, b)`.
- Reissner–Nordström / Kerr / Kerr–Newman are parked (see the R2 roadmap).

## Methods

### is_causal

```python
def is_causal(self, u, v) -> bool
```

`True` iff `v` is in the future of `u` under the radial null condition `Δt ≥ |Δr*|`.

## Example

```python
import pycauset as pc
from pycauset import spacetime

c = pc.causet(n=200, spacetime=spacetime.Schwarzschild(mass=1.0), seed=1)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.DeSitter.md|DeSitter]]
- [[guides/Spacetime.md|Spacetime guide]]
