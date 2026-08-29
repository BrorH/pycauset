# pycauset.spacetime.FLRW

```python
class FLRW(dimension=2, scale_factor=0, time_extent=1.0, space_extent=1.0)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

FLRW spacetime with flat spatial slices (k=0): `ds² = −dt² + a(t)² dx⃗²`.

## Parameters

*   **dimension** (*int*): Spacetime dimension `d` (≥ 2).
*   **scale_factor** (*float | callable*): A power-law exponent `p` (`a(t) = t^p`; `p=0` is Minkowski) or a callable `a(t) -> float`.
*   **time_extent** (*float*): The time range.
*   **space_extent** (*float*): The spatial box size.

## Notes (honest caveats)

- `is_causal` uses the null condition `∫_{t₁}^{t₂} dt/a(t) ≥ |Δx⃗|`.
- The **sampler** is uniform in `(t, x⃗)`, a documented parametrization, not the FLRW-invariant measure unless `a(t)` is constant.
- `scalar_coeffs` raises `NotImplementedError`, coefficients are manual `(a, b)`.

## Methods

### is_causal

```python
def is_causal(self, u, v) -> bool
```

`True` iff the comoving horizon between `u` and `v` covers their spatial separation.

### sample / volume

```python
def sample(self, rng, n) -> np.ndarray
def volume(self) -> float
```

`volume` is the finite-box volume `L^{d-1} · ∫ a(t)^{d-1} dt`.

## Example

```python
from pycauset import spacetime

# a(t) = t  (radiation-like), or pass a callable a(t)
st = spacetime.FLRW(2, scale_factor=1, time_extent=1.0, space_extent=1.0)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.DeSitter.md|DeSitter]]
- [[docs/classes/spacetime/pycauset.spacetime.AntiDeSitter.md|AntiDeSitter]]
- [[guides/Spacetime.md|Spacetime guide]]
