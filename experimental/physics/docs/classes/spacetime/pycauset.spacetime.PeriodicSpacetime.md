# pycauset.spacetime.PeriodicSpacetime

```python
class PeriodicSpacetime(base: Spacetime, periods: dict | float, max_images: int = 3)
```

Inherits from: [[docs/classes/spacetime/pycauset.spacetime.Spacetime.md|Spacetime]]

Wraps a spacetime with periodic identification along chosen axes: points are identified `x[a] ~ x[a] + L` and live in the fundamental domain `[0, L)` along each periodic axis.

## Parameters

*   **base** (*Spacetime*): The spacetime to wrap.
*   **periods** (*dict | float*): Maps an axis index to its period `L > 0`. A bare number is shorthand for "wrap every spacelike axis (index `>= 1`) with that period".
*   **max_images** (*int, optional*): The causal predicate checks periodic images within `±max_images` shifts per axis. Keep it `>= (time extent)/(period)` for exactness on a bounded base.

## Methods

### volume

```python
def volume(self) -> float
```

The base volume (the fundamental domain).

### sample

```python
def sample(self, rng, n) -> np.ndarray
```

Draws `n` base samples and wraps each periodic axis into `[0, L)`.

### is_causal

```python
def is_causal(self, u, v) -> bool
```

The quotient order: `u ≺ v` iff *some* periodic image of `v` (within `±max_images`) lies in the future of `u`.

> **Caveat:** only **spacelike** axes may be periodic. Periodic time would produce closed timelike curves, so axis `0` raises `NotImplementedError` rather than silently shipping a pathological order.

## Example

```python
from pycauset import spacetime

box = spacetime.MinkowskiBox(2, 10.0, 10.0)
# Identify the spatial direction with period 5 → a spatial circle.
cylinder = spacetime.PeriodicSpacetime(box, periods={1: 5.0})
```

## See also

- [[docs/classes/spacetime/pycauset.spacetime.RestrictedSpacetime.md|RestrictedSpacetime]]
- [[docs/classes/spacetime/pycauset.spacetime.ConformalSpacetime.md|ConformalSpacetime]]
- [[docs/classes/spacetime/pycauset.spacetime.MinkowskiCylinder.md|MinkowskiCylinder]]
