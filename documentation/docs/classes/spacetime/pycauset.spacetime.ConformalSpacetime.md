# pycauset.spacetime.ConformalSpacetime

```python
class ConformalSpacetime(base: Spacetime, conformal_factor: Callable, volume: float | None = None, max_weight: float | None = None)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Wraps a spacetime with a position-dependent conformal factor `Omega(x)`, rescaling the metric by `Omega(x)^2`. Conformal transformations preserve the **causal light-cone**, so `is_causal` is inherited verbatim, but they rescale the volume measure by `Omega^d`.

## Parameters

*   **base** (*Spacetime*): The spacetime to wrap.
*   **conformal_factor** (*callable*): `Omega(coords) -> float`, strictly positive on the base's support. Points where it returns `<= 0` are rejected.
*   **volume** (*float, optional*): The conformal volume `∫ Omega^d dV`. If omitted, it is Monte-Carlo estimated as `E[Omega^d] * V_base` so `volume ↔ sample` stay consistent.
*   **max_weight** (*float, optional*): An upper bound on `Omega^d`, used as the rejection bound in `sample`. If omitted, it is calibrated by sampling the base. Pass it explicitly when the factor has sharp, poorly-sampled peaks.

## Methods

### volume

```python
def volume(self) -> float
```

The conformal volume (provided, or Monte-Carlo estimated).

### sample

```python
def sample(self, rng, n) -> np.ndarray
```

Draws `n` points by rejection-sampling the base with weight `Omega^d`.

### is_causal

```python
def is_causal(self, u, v) -> bool
```

Inherited unchanged from `base` (conformal maps preserve the light cone).

## Example

```python
from pycauset import spacetime

box = spacetime.MinkowskiBox(2, 10.0, 10.0)
# Omega = 2 rescales the volume measure by 2^2 = 4.
blown_up = spacetime.ConformalSpacetime(box, conformal_factor=lambda c: 2.0)
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.RestrictedSpacetime.md|RestrictedSpacetime]]
- [[docs/classes/spacetime/pycauset.spacetime.TransformedSpacetime.md|TransformedSpacetime]]
- [[docs/classes/spacetime/pycauset.spacetime.PeriodicSpacetime.md|PeriodicSpacetime]]
